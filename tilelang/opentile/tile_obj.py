"""Tile backend stage-1: lightweight .o load + launch.

Device compilation happens outside the repository: users hand-compile the
MLIR/CCE kernel into a relocatable or executable CCE ELF (hereafter ``.o``)
and pass the file path to :func:`compile_tile_obj` or :func:`load_tile_obj`.

This module implements the runtime half only:

1. :func:`extract_launch_info` walks the lowered TIR and extracts
   - the kernel name (``global_symbol``),
   - the argument types (buffer params -> "handle", scalars -> dtype),
   - the launch grid expression (``thread_extent`` / ``blockIdx.x``),
2. :class:`TileObjKernel` evaluates the launch grid with concrete tensor
   shapes, passes an optional additional dynamic UBUF size, encodes arguments,
   and submits through ``tl.tile.LaunchKernelWithBinaryKind``.

Notes / contract with the hand-compiled ``.o``:
- A single-core kernel symbol must equal the IR ``global_symbol``.
- For a mixed kernel, the caller must provide the base runtime entry name
  rather than an individual ``_mix_aic`` or ``_mix_aiv`` symbol.
- The kernel argument list (order + types) must match the IR params.
- ``grid_override`` overrides the grid extracted from TIR.
- ``ubuf_override`` is the additional dynamic UBUF size passed at launch.
  Compiler-allocated static UB comes from the object metadata; CANN Runtime
  combines the static and dynamic portions.
"""

from __future__ import annotations

import dataclasses
import re
import struct
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tilelang import tvm
from tvm import tirx

if TYPE_CHECKING:
    pass

__all__ = ["LaunchInfo", "TileObjKernel", "compile_tile_obj", "extract_launch_info", "load_tile_obj"]


def _torch_module():
    """Lazily import torch.

    Everything compiler-side (extract_launch_info / compile_tile_obj
    metadata handling) must work without torch installed; only the runtime
    launch path (TileObjKernel.__call__) needs it to read tensor data
    pointers.
    """
    import torch

    return torch


# dtype string -> ACL argument kind (see src/tile/runtime/tile_obj_launcher.cc)
_SCALAR_DTYPE_TO_KIND = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "bool": "int32",
}

_BINARY_KIND_ALIASES = {
    "auto": "auto",
    "aiv": "aiv",
    "vector": "aiv",
    "aic": "aicore",
    "aicore": "aicore",
    "mix": "aicore",
    "aicube": "aicube",
    "cube": "aicube",
}

def _normalize_binary_kind(binary_kind: str) -> str:
    if not isinstance(binary_kind, str):
        raise TypeError(f"binary_kind must be a string, got {type(binary_kind).__name__}")
    normalized = _BINARY_KIND_ALIASES.get(binary_kind.lower())
    if normalized is None:
        choices = ", ".join(sorted(_BINARY_KIND_ALIASES))
        raise ValueError(f"unsupported binary_kind `{binary_kind}`; expected one of {choices}")
    return normalized

@dataclasses.dataclass(frozen=True)
class DynamicSymbolSource:
    """Runtime source of a dynamic TIR variable."""

    kind: str   # "scalar", "shape", "stride"
    param_index: int
    dim_index: int = -1
    scale: int = 1

@dataclasses.dataclass
class LaunchInfo:
    """Metadata extracted from the lowered TIR of a Tile kernel."""

    name: str
    arg_types: list[str]
    # Per handle argument: the shape expressions of the corresponding buffer.
    handle_shapes: list[list[tirx.PrimExpr]]
    # Per handle argument: the normalized dtype name of the corresponding
    # buffer (e.g. "float16", "bfloat"); checked against tensor.dtype at
    # launch time (TODO 2).
    handle_dtypes: list[str]
    # Grid: one extent per blockIdx dimension; the launch grid is the product.
    grid_exprs: list[tirx.PrimExpr]
    #Runtime sources used to resolve dynamic variables in grid expressions.
    dynamic_symbol_sources: dict[str, list[DynamicSymbolSource]] = (
        dataclasses.field(default_factory=dict)
    )
    # Trailing i32 grid params (gridX, gridY, gridZ) injected when grid_dims
    # is set; None otherwise.
    trailing_grid_dims: tuple[int, int, int] | None = None

def _add_dynamic_symbol_source(
        sources: dict[str, list[DynamicSymbolSource]],
        var: tirx.Var,
        source: DynamicSymbolSource,
) -> None:
    candidates = sources.setdefault(var.name, [])
    if source not in candidates:
        candidates.append(source)

def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    match = re.match(r"^(u?int|float|bool|bfloat|fp\d*e?\w*|float\d*e?\w*)(\d+)?", text)
    if match is None:
        return text
    base = match.group(1)
    bits = match.group(2)
    if base in ("int", "uint", "float"):
        return f"{base}{bits}"
    return text


def _iter_stmt_fields(node: Any):
    """Yield the child statements of common tirx statement nodes."""
    # SBlockRealize wraps its SBlock in `block` (not `body`); SBlock has both
    # `body` and `init`.  Cover all common child-bearing fields defensively.
    for attr in ("body", "block", "init", "then_case", "else_case"):
        child = getattr(node, attr, None)
        if child is not None:
            yield child


def _collect_grid(
    stmt: Any,
    grid_exprs: list[tirx.PrimExpr],
    seen: set[int],
) -> None:
    """Walk the statement tree and collect blockIdx extents."""
    if id(stmt) in seen:
        return
    seen.add(id(stmt))

    if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "thread_extent":
        iter_var = stmt.node
        tag = getattr(iter_var, "thread_tag", "")
        if str(tag).startswith("blockIdx"):
            grid_exprs.append(stmt.value)

    for child in _iter_stmt_fields(stmt):
        _collect_grid(child, grid_exprs, seen)


def extract_launch_info(lowered_mod: Any, expect_name: str | None = None) -> LaunchInfo:
    """Extract launch metadata from a lowered (tile-pipeline) IRModule.

    Parameters
    ----------
    lowered_mod : IRModule
        Result of ``tilelang.lower(func, target="tile")`` — for the tile
        backend this is the whole lowered module (host/device are not split).
    expect_name : str, optional
        The ``global_symbol`` of the original PrimFunc.  Used to pick the
        main function when the pipeline emitted wrappers.
    """
    funcs = list(lowered_mod.functions.values())
    if not funcs:
        raise ValueError("extract_launch_info: the lowered module has no functions")

    # Prefer the function carrying global_symbol; fall back to the first one.
    main_func = None
    for func in funcs:
        global_symbol = func.attrs.get("global_symbol", None) if hasattr(func, "attrs") else None
        if expect_name is not None:
            if global_symbol is not None and str(global_symbol) == expect_name:
                main_func = func
                break
        elif global_symbol is not None:
            main_func = func
            break
    if main_func is None:
        main_func = funcs[0]

    name = str(main_func.attrs["global_symbol"]) if "global_symbol" in main_func.attrs else str(list[Any](lowered_mod.functions.keys())[0])

    # Argument types and per-handle shapes/dtypes, in param order.
    arg_types: list[str] = []
    handle_shapes: list[list[tirx.PrimExpr]] = []
    handle_dtypes: list[str] = []
    dynamic_symbol_sources: dict[str, list[DynamicSymbolSource]] = {}

    buffer_map = getattr(main_func, "buffer_map", {})

    for param_index, param in enumerate(main_func.params):
        if param in buffer_map:
            buffer = buffer_map[param]

            arg_types.append("handle")
            handle_shapes.append(list(buffer.shape))
            handle_dtypes.append(_dtype_name(buffer.dtype))

            for dim_index, shape in enumerate(buffer.shape):
                if isinstance(shape, tirx.Var):
                    _add_dynamic_symbol_source(
                        dynamic_symbol_sources,
                        shape,
                        DynamicSymbolSource(
                            kind="shape",
                            param_index=param_index,
                            dim_index=dim_index,
                        ),
                    )

            for dim_index, stride in enumerate(buffer.strides or []):
                if not isinstance(stride, tirx.Var):
                    continue

                element_bits = buffer.dtype.bits * buffer.dtype.lanes
                stride_scale = 8 // element_bits if element_bits < 8 else 1

                _add_dynamic_symbol_source(
                    dynamic_symbol_sources,
                    stride,
                    DynamicSymbolSource(
                        kind="stride",
                        param_index=param_index,
                        dim_index=dim_index,
                        scale=stride_scale,
                    ),
                )
        else:
            kind = _SCALAR_DTYPE_TO_KIND.get(_dtype_name(param.dtype))
            if kind is None:
                raise ValueError(
                    "extract_launch_info: unsupported scalar dtype "
                    f"`{param.dtype}` for parameter `{param}`"
                )

            arg_types.append(kind)

            if isinstance(param, tirx.Var):
                _add_dynamic_symbol_source(
                    dynamic_symbol_sources,
                    param,
                    DynamicSymbolSource(
                        kind="scalar",
                        param_index=param_index,
                    ),
                )

    grid_exprs: list[tirx.PrimExpr] = []
    _collect_grid(main_func.body, grid_exprs, set())

    if not grid_exprs:
        raise ValueError("extract_launch_info: no blockIdx thread_extent found; is this a T.Kernel function lowered with target='tile'?")

    return LaunchInfo(
        name=name,
        arg_types=arg_types,
        handle_shapes=handle_shapes,
        handle_dtypes=handle_dtypes,
        grid_exprs=grid_exprs,
        dynamic_symbol_sources=dynamic_symbol_sources,
    )


def _eval_expr(expr: tirx.PrimExpr, var_map: dict[str, int]) -> int:
    """Evaluate a small integer expression against a symbol table."""
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    if isinstance(expr, tirx.Var):
        key = getattr(expr, "name", str(expr))
        if key not in var_map:
            raise KeyError(f"unknown symbol `{key}` in launch expression; pass grid_override/ubuf_override or precompute the value")
        return var_map[key]
    if isinstance(expr, tirx.Cast):
        return _eval_expr(expr.value, var_map)
    if isinstance(expr, tirx.Add):
        return _eval_expr(expr.a, var_map) + _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.Sub):
        return _eval_expr(expr.a, var_map) - _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.Mul):
        return _eval_expr(expr.a, var_map) * _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.FloorDiv):
        return _eval_expr(expr.a, var_map) // _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.FloorMod):
        return _eval_expr(expr.a, var_map) % _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.Mod):
        return _eval_expr(expr.a, var_map) % _eval_expr(expr.b, var_map)
    if isinstance(expr, tirx.Min):
        return min(_eval_expr(expr.a, var_map), _eval_expr(expr.b, var_map))
    if isinstance(expr, tirx.Max):
        return max(_eval_expr(expr.a, var_map), _eval_expr(expr.b, var_map))
    raise TypeError(
        f"unsupported expression node {type(expr).__name__} in launch expression; pass grid_override/ubuf_override or precompute the value"
    )


def _current_npu_stream() -> int:
    """Raw ACL stream handle of torch's current NPU stream, or 0 (default).

    Reuses ``BaseKernelAdapter.get_current_stream_functor`` so the NPU/CUDA
    branch logic lives in exactly one place (mirrors tilelang-ascend-cce).
    """
    try:
        from tilelang.jit.adapter.base import BaseKernelAdapter

        return int(BaseKernelAdapter.get_current_stream_functor()())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Launch-time argument validation (TODO 2: wrong dtype must not fail silently)
# ---------------------------------------------------------------------------

# torch dtype names (after stripping the "torch." prefix) that differ from the
# normalized tilelang dtype names.
_TORCH_DTYPE_ALIASES = {"bfloat16": "bfloat"}


def _dtype_family_bits(name: str) -> tuple[str, int]:
    """Return the TVM dtype family and per-lane bit width."""
    try:
        dtype = tvm.DataType(name)
    except ValueError:
        return name, 0

    families = {
        tvm.DataTypeCode.INT: "int",
        tvm.DataTypeCode.UINT: "uint",
        tvm.DataTypeCode.FLOAT: "float",
        tvm.DataTypeCode.BFLOAT: "bfloat",
        tvm.DataTypeCode.BOOL: "bool",
    }
    family = families.get(dtype.type_code)
    return (family, dtype.bits) if family is not None else (name, 0)


def _check_handle_dtype(arg_index: int, torch_dtype: str, expected: str) -> None:
    """Check one tensor's dtype against the IR buffer dtype.

    ``torch_dtype`` is ``str(tensor.dtype)`` with the ``torch.`` prefix
    stripped (e.g. "float16").  Same-width int/uint pairs are accepted: they
    share the bit pattern, only the kernel-side interpretation differs.
    Unknown expected dtypes downgrade to a warning (cannot verify reliably).
    """
    got = _TORCH_DTYPE_ALIASES.get(torch_dtype, torch_dtype)
    if got == expected:
        return
    got_fb, expected_fb = _dtype_family_bits(got), _dtype_family_bits(expected)
    if got_fb == expected_fb:
        return
    if {got_fb[0], expected_fb[0]} <= {"int", "uint"} and got_fb[1] == expected_fb[1] != 0:
        return
    if expected_fb[1] == 0:
        warnings.warn(
            f"argument {arg_index}: cannot verify tensor dtype `{torch_dtype}` against unknown IR buffer dtype `{expected}`",
            stacklevel=3,
        )
        return
    raise TypeError(
        f"argument {arg_index} dtype mismatch: kernel expects `{expected}` but got "
        f"`{torch_dtype}`; convert with tensor.to(...) -- the launch ABI passes a raw "
        f"pointer, a wrong dtype computes silently wrong results"
    )


def _encode_scalar_arg(kind: str, value: Any) -> int:
    """Encode a scalar argument into the int64 representation used by the launcher."""
    if kind == "float32":
        return struct.unpack("<i", struct.pack("<f", value))[0]
    if kind == "float64":
        return struct.unpack("<q", struct.pack("<d", value))[0]
    return int(value)


class TileObjKernel:
    """Callable wrapper around a hand-compiled Tile object file."""

    def __init__(
        self,
        info: LaunchInfo,
        obj_bytes: bytes,
        grid_override: int | None = None,
        ubuf_override: int | None = None,
        binary_kind: str = "auto",
    ):
        self.info = info
        self.obj_bytes = obj_bytes
        self.grid_override = grid_override
        self.ubuf_override = 0 if ubuf_override is None else ubuf_override
        self.binary_kind = _normalize_binary_kind(binary_kind)
        self.trailing_grid_dims = tuple(info.trailing_grid_dims) if info.trailing_grid_dims else None
        self._launch = tvm.ffi.get_global_func("tl.tile.LaunchKernelWithBinaryKind")
        if self._launch is None:
            raise RuntimeError(
                "tl.tile.LaunchKernelWithBinaryKind is not registered; rebuild tilelang with src/tile enabled "
                "(see src/tile/CMakeLists.txt)"
            )

    def _resolve_dynamic_symbol(
        self,
        name: str,
        args: list[Any],
    ) -> int:
        candidates = self.info.dynamic_symbol_sources.get(name, [])
        unavailable: list[str] = []

        for source in candidates:
            if source.param_index >= len(args):
                unavailable.append(
                    f"{source.kind} source uses missing argument "
                    f"{source.param_index}"
                )
                continue

            value = args[source.param_index]

            if source.kind == "scalar":
                try:
                    return int(value)
                except (TypeError, ValueError):
                    unavailable.append(
                        f"argument {source.param_index} is not a scalar"
                    )
                    continue

            if source.kind == "shape":
                try:
                    return int(value.shape[source.dim_index])
                except (AttributeError, IndexError, TypeError):
                    unavailable.append(
                        f"argument {source.param_index} has no shape dimension "
                        f"{source.dim_index}"
                    )
                    continue
            if source.kind == "stride":
                try:
                    stride = value.stride()[source.dim_index]
                    return int(stride) * source.scale
                except (AttributeError, IndexError, TypeError):
                    unavailable.append(
                        f"argument {source.param_index} has no stride dimension "
                        f"{source.dim_index}"
                    )
                    continue

            unavailable.append(f"unknown source kind: {source.kind}")

        details = "; ".join(unavailable) or "no runtime source was recorded"
        raise ValueError(
            f"cannot resolve dynamic grid variable `{name}`: {details}"
        )

    def _build_var_map(self, args: list[Any]) -> dict[str, int]:
        """Resolve grid variables from scalar, tensor shape and stride arguments."""
        return {
            name: self._resolve_dynamic_symbol(name, args)
            for name in self.info.dynamic_symbol_sources
        }

    def _compute_grid(self, args: list[Any]) -> int:
        if self.grid_override is not None:
            return self.grid_override
        var_map = self._build_var_map(args)
        grid = 1
        for expr in self.info.grid_exprs:
            grid *= _eval_expr(expr, var_map)
        if grid <= 0:
            raise ValueError(f"computed grid {grid} is not positive; check the runtime shapes and scalar arguments")
        return grid

    def _compute_ubuf(self) -> int:
        return self.ubuf_override

    def _encode_args(self, args: list[Any]) -> list[int]:
        torch = _torch_module()
        encoded: list[int] = []
        handle_index = 0
        for i, kind in enumerate(self.info.arg_types):
            value = args[i]
            if kind == "handle":
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"argument {i} expects a torch.Tensor (handle), got {type(value)}")
                # Dtype first (pairing correctness), then device/contiguity
                # (runtime usability) -- a dtype mismatch computes silently
                # wrong results even on a valid NPU tensor.
                _check_handle_dtype(
                    i,
                    str(value.dtype).split(".")[-1],
                    self.info.handle_dtypes[handle_index],
                )
                if value.device.type != "npu":
                    raise ValueError(
                        f"argument {i} is on {value.device}, expected an NPU tensor; the tile launch ABI passes a raw device pointer"
                    )
                if not value.is_contiguous():
                    raise ValueError(f"argument {i} must be contiguous; the tile launch ABI passes a raw data pointer without strides")
                encoded.append(int(value.data_ptr()))
                handle_index += 1
            else:
                encoded.append(_encode_scalar_arg(kind, value))
        return encoded

    def __call__(self, *args: Any) -> None:
        trailing = list(self.trailing_grid_dims or ())
        user_arg_count = len(self.info.arg_types) - len(trailing)
        if len(args) != user_arg_count:
            raise ValueError(
                f"kernel `{self.info.name}` expects {user_arg_count} arguments "
                f"({len(trailing)} trailing grid params are injected automatically) "
                f"but got {len(args)}"
            )
        arg_list = list(args)
        grid = self._compute_grid(arg_list)
        ubuf = self._compute_ubuf()
        arg_list.extend(trailing)
        encoded = self._encode_args(arg_list)
        stream = _current_npu_stream()
        self._launch(self.obj_bytes, self.info.name, grid, ubuf, stream, self.info.arg_types, encoded, self.binary_kind)


def compile_tile_obj(
    func_or_artifact,
    obj_path: str | Path,
    grid_override: int | None = None,
    ubuf_override: int | None = None,
    binary_kind: str = "auto",
) -> TileObjKernel:
    """Bind a hand-compiled object to TileLang kernel metadata.

    Parameters
    ----------
    func_or_artifact : PrimFunc | CompiledArtifact | IRModule
        The kernel, in one of three accepted forms:

        - ``PrimFunc``: lowered internally via ``tilelang.lower(func,
          target="tile")`` — convenient when you only have the DSL result.
        - ``CompiledArtifact``: the result of an earlier
          ``tilelang.lower(..., target="tile")`` call — reuse it when you
          already lowered once to obtain the MLIR for hand-compilation.
        - ``IRModule``: a lowered tile module (e.g. ``artifact.device_mod``).
    obj_path : str or Path
        Path to the hand-compiled CCE object (relocatable or executable ELF).
    grid_override : int, optional
        Skip TIR grid extraction and always launch with this block count.
    ubuf_override : int, optional
        Additional dynamic UBUF size in bytes passed through the ACL launch
        attribute. CANN Runtime adds it to the compiler-allocated static UB
        recorded in the object metadata. When omitted, zero is passed, which
        is the normal setting for statically planned OpenTileAS kernels.
    binary_kind : str, optional
        ACL object kind. Use ``"aiv"`` for vector-only objects and
        ``"aicore"`` for generic or mixed AIC/AIV objects. ``"auto"`` keeps
        the ACL default behavior.
    """
    # No top-level torch import here: compile_tile_obj only inspects already
    # lowered artifacts/modules, so it must not pull in tilelang.engine.param
    # (whose import chain reaches torch).  The PrimFunc branch below lowers
    # lazily, which is the only path that legitimately needs the full engine.
    expect_name = None
    if hasattr(func_or_artifact, "device_mod") and hasattr(func_or_artifact, "host_mod"):
        # CompiledArtifact (duck-typed to avoid the torch-importing param module)
        lowered_mod = func_or_artifact.device_mod
    elif hasattr(func_or_artifact, "body") and hasattr(func_or_artifact, "params"):
        # PrimFunc
        from tilelang.engine.lower import lower as _tl_lower

        artifact = _tl_lower(func_or_artifact, target="tile")
        lowered_mod = artifact.device_mod
        attrs = func_or_artifact.attrs
        if "global_symbol" in attrs:
            expect_name = str(attrs["global_symbol"])
    elif hasattr(func_or_artifact, "functions"):
        # IRModule (lowered tile module, e.g. artifact.device_mod)
        lowered_mod = func_or_artifact
    else:
        raise TypeError(f"compile_tile_obj expects a PrimFunc, CompiledArtifact or IRModule, got {type(func_or_artifact).__name__}")

    info = extract_launch_info(lowered_mod, expect_name=expect_name)

    if ubuf_override is not None:
        if not isinstance(ubuf_override, int) or ubuf_override < 0:
            raise ValueError(
                "ubuf_override must be a non-negative integer, "
                f"got {ubuf_override}"
            )
    runtime_ubuf_size = 0 if ubuf_override is None else ubuf_override
    obj_bytes = Path(obj_path).read_bytes()

    normalized_binary_kind = _normalize_binary_kind(binary_kind)

    return TileObjKernel(
        info,
        obj_bytes,
        grid_override=grid_override,
        ubuf_override=runtime_ubuf_size,
        binary_kind=normalized_binary_kind,
    )

def load_tile_obj(
    obj_path: str | Path,
    kernel_name: str,
    arg_types: list[str] | tuple[str, ...],
    grid: int,
    ubuf_size: int = 0,
    binary_kind: str = "auto",
    handle_dtypes: list[str] | tuple[str, ...] | None = None,
    grid_dims: tuple[int, int, int] | None = None,
) -> TileObjKernel:
    """Bind a raw CCE object using an explicit launch manifest.

    Unlike :func:`compile_tile_obj`, this entry point does not require a lowered TileLang artifact. 
    Kernel name, argument ABI, grid, additional dynamic UBUF, and binary kind 
    must come from a trusted manifest such as onboard's ``runner.json``; 
    they cannot be recovered reliably from a stripped ``.o``.
    ``handle_dtypes`` should list one dtype per ``handle`` argument; omitted
    dtypes are treated as unknown and only produce a launch-time warning.
    ``grid_dims``, when given as ``(gridX, gridY, gridZ)`` appends three
    trailing i32 grid arguments to the device ABI —— the shape OpenTile/Triton
    kernels expect from the inject-grid-params pass.
    """
    if not isinstance(kernel_name, str) or not kernel_name:
        raise ValueError("kernel_name must be a non-empty string")
    if not isinstance(grid, int) or grid <= 0:
        raise ValueError(f"grid must be a positive integer, got {grid}")
    if not isinstance(ubuf_size, int) or ubuf_size < 0:
        raise ValueError(f"ubuf_size must be a non-negative integer, got {ubuf_size}")

    if grid_dims is not None:
        grid_dims = tuple(int(d) for d in grid_dims)
        if len(grid_dims) != 3 or any(d <= 0 for d in grid_dims):
            raise ValueError(f"grid_dims must be three positive ints (gridX, gridY, gridZ), got {grid_dims}")

    arg_types = list(arg_types)
    supported = {"handle", *_SCALAR_DTYPE_TO_KIND.values()}
    unsupported = [kind for kind in arg_types if kind not in supported]
    if unsupported:
        raise ValueError(f"unsupported tile object argument type(s): {unsupported}")
    handle_count = arg_types.count("handle")
    if handle_dtypes is None:
        handle_dtypes = ["unknown"] * handle_count
    else:
        handle_dtypes = [_dtype_name(dtype) for dtype in handle_dtypes]
        if len(handle_dtypes) != handle_count:
            raise ValueError(f"handle_dtypes has {len(handle_dtypes)} entries but arg_types contains {handle_count} handles")

    if grid_dims is not None:
        arg_types.extend(["int32", "int32", "int32"])

    obj_bytes = Path(obj_path).read_bytes()
    normalized_binary_kind = _normalize_binary_kind(binary_kind)

    info = LaunchInfo(
        name=kernel_name,
        arg_types=arg_types,
        handle_shapes=[[] for kind in arg_types if kind == "handle"],
        handle_dtypes=handle_dtypes,
        grid_exprs=[],
        trailing_grid_dims=grid_dims,
    )

    return TileObjKernel(
        info,
        obj_bytes,
        grid_override=grid,
        ubuf_override=ubuf_size,
        binary_kind=normalized_binary_kind,
    )
