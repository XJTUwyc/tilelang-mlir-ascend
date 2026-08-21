"""Tile backend stage-1: lightweight .o load + launch.

Device compilation happens outside the repository: users hand-compile the
MLIR/CCE kernel into a relocatable or executable CCE ELF (hereafter ``.o``)
and pass the file path to :func:`compile_tile_obj`.

This module implements the runtime half only:

1. :func:`extract_launch_info` walks the lowered TIR and extracts
   - the kernel name (``global_symbol``),
   - the argument types (buffer params -> "handle", scalars -> dtype),
   - the launch grid expression (``thread_extent`` / ``blockIdx.x``),
   - the dynamic UBUF size (sum over ``shared.dyn`` allocations).
2. :class:`TileObjKernel` evaluates grid/UBUF with the concrete tensor
   shapes, encodes arguments as int64, grabs the current NPU stream and
   submits via the C++ entry point ``tl.tile.LaunchKernel``.

Notes / contract with the hand-compiled ``.o``:
- The kernel symbol in the ``.o`` must equal the IR ``global_symbol``.
- The kernel argument list (order + types) must match the IR params.
- ``grid_override`` / ``ubuf_override`` are escape hatches when the TIR
  expressions are too complex for the built-in evaluator.
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

__all__ = ["LaunchInfo", "TileObjKernel", "compile_tile_obj", "extract_launch_info"]


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


@dataclasses.dataclass
class LaunchInfo:
    """Metadata extracted from the lowered TIR of a Tile kernel."""

    name: str
    arg_types: list[str]
    # Per handle argument: the shape expressions of the corresponding buffer.
    handle_shapes: list[list[tirx.PrimExpr]]
    # Grid: one extent per blockIdx dimension; the launch grid is the product.
    grid_exprs: list[tirx.PrimExpr]
    # Dynamic UBUF bytes as a PrimExpr, or None when the kernel needs none.
    ubuf_expr: tirx.PrimExpr | None


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


def _buffer_bytes(dtype: Any) -> int:
    text = _dtype_name(dtype)
    match = re.search(r"(\d+)", text)
    if match is None:
        return 1
    return max(1, int(match.group(1)) // 8)


def _iter_stmt_fields(node: Any):
    """Yield the child statements of common tirx statement nodes."""
    # SBlockRealize wraps its SBlock in `block` (not `body`); SBlock has both
    # `body` and `init`.  Cover all common child-bearing fields defensively.
    for attr in ("body", "block", "init", "then_case", "else_case"):
        child = getattr(node, attr, None)
        if child is not None:
            yield child


def _collect_grid_ubuf(
    stmt: Any,
    grid_exprs: list[tirx.PrimExpr],
    ubuf_terms: list[tirx.PrimExpr],
    seen: set[int],
) -> None:
    """Walk the statement tree collecting blockIdx extents and shared.dyn sizes."""
    if id(stmt) in seen:
        return
    seen.add(id(stmt))

    if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "thread_extent":
        iter_var = stmt.node
        tag = getattr(iter_var, "thread_tag", "")
        if str(tag).startswith("blockIdx"):
            grid_exprs.append(stmt.value)
    elif isinstance(stmt, tirx.AllocBuffer):
        buffer = stmt.buffer
        if "shared.dyn" in str(buffer.scope()):
            size = tirx.IntImm("int64", 1)
            for extent in buffer.shape:
                size = tirx.Mul(size, tirx.Cast("int64", extent))
            ubuf_terms.append(tirx.Mul(size, tirx.IntImm("int64", _buffer_bytes(buffer.dtype))))
    elif isinstance(stmt, tirx.SBlock):
        # In tile lowered IR the shared allocations live on the SBlock's
        # alloc_buffers list (sblock_alloc_buffer), not as AllocBuffer
        # statements.  The AllocBuffer branch above is kept as a fallback.
        for buffer in stmt.alloc_buffers or []:
            if "shared.dyn" in str(buffer.scope()):
                size = tirx.IntImm("int64", 1)
                for extent in buffer.shape:
                    size = tirx.Mul(size, tirx.Cast("int64", extent))
                ubuf_terms.append(tirx.Mul(size, tirx.IntImm("int64", _buffer_bytes(buffer.dtype))))

    for child in _iter_stmt_fields(stmt):
        _collect_grid_ubuf(child, grid_exprs, ubuf_terms, seen)


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

    name = str(main_func.attrs["global_symbol"]) if "global_symbol" in main_func.attrs else str(list(lowered_mod.functions.keys())[0])

    # Argument types and per-handle shapes, in param order.
    arg_types: list[str] = []
    handle_shapes: list[list[tirx.PrimExpr]] = []
    buffer_map = getattr(main_func, "buffer_map", {})
    for param in main_func.params:
        if param in buffer_map:
            arg_types.append("handle")
            handle_shapes.append([expr for expr in buffer_map[param].shape])
        else:
            kind = _SCALAR_DTYPE_TO_KIND.get(_dtype_name(param.dtype))
            if kind is None:
                raise ValueError(f"extract_launch_info: unsupported scalar dtype `{param.dtype}` for parameter `{param}`")
            arg_types.append(kind)
            # Scalars do not occupy a handle_shapes slot: handle_shapes is
            # indexed by the handle counter, not by the parameter index.

    grid_exprs: list[tirx.PrimExpr] = []
    ubuf_terms: list[tirx.PrimExpr] = []
    _collect_grid_ubuf(main_func.body, grid_exprs, ubuf_terms, set())

    if not grid_exprs:
        raise ValueError("extract_launch_info: no blockIdx thread_extent found; is this a T.Kernel function lowered with target='tile'?")

    ubuf_expr: tirx.PrimExpr | None = None
    if ubuf_terms:
        ubuf_expr = ubuf_terms[0]
        for term in ubuf_terms[1:]:
            ubuf_expr = tirx.Add(ubuf_expr, term)

    return LaunchInfo(
        name=name,
        arg_types=arg_types,
        handle_shapes=handle_shapes,
        grid_exprs=grid_exprs,
        ubuf_expr=ubuf_expr,
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


class TileObjKernel:
    """Callable wrapper around a hand-compiled Tile object file."""

    def __init__(
        self,
        info: LaunchInfo,
        obj_bytes: bytes,
        grid_override: int | None = None,
        ubuf_override: int | None = None,
    ):
        self.info = info
        self.obj_bytes = obj_bytes
        self.grid_override = grid_override
        self.ubuf_override = ubuf_override
        self._launch = tvm.ffi.get_global_func("tl.tile.LaunchKernel")
        if self._launch is None:
            raise RuntimeError(
                "tl.tile.LaunchKernel is not registered; rebuild tilelang with src/tile enabled (see src/tile/CMakeLists.txt)"
            )

    def _build_var_map(self, args: list[Any]) -> dict[str, int]:
        """Bind plain-Var shape dims to the concrete tensor shapes."""
        torch = _torch_module()
        var_map: dict[str, int] = {}
        handle_index = 0
        for i, kind in enumerate(self.info.arg_types):
            if kind != "handle":
                continue
            tensor = args[i]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"argument {i} expects a torch.Tensor (handle), got {type(tensor)}")
            shapes = self.info.handle_shapes[handle_index]
            if len(shapes) != tensor.dim():
                raise ValueError(f"argument {i} expects rank {len(shapes)} but got {tensor.dim()}")
            for expr, value in zip(shapes, tensor.shape):
                if isinstance(expr, tirx.Var):
                    var_map[getattr(expr, "name", str(expr))] = int(value)
            handle_index += 1
        return var_map

    def _compute_grid(self, args: list[Any]) -> int:
        if self.grid_override is not None:
            return self.grid_override
        var_map = self._build_var_map(args)
        grid = 1
        for expr in self.info.grid_exprs:
            grid *= _eval_expr(expr, var_map)
        if grid <= 0:
            raise ValueError(f"computed grid {grid} is not positive; check the kernel shapes")
        return grid

    def _compute_ubuf(self, args: list[Any]) -> int:
        if self.ubuf_override is not None:
            return self.ubuf_override
        if self.info.ubuf_expr is None:
            return 0
        return _eval_expr(self.info.ubuf_expr, self._build_var_map(args))

    def _encode_args(self, args: list[Any]) -> list[int]:
        torch = _torch_module()
        encoded: list[int] = []
        for i, kind in enumerate(self.info.arg_types):
            value = args[i]
            if kind == "handle":
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"argument {i} expects a torch.Tensor (handle), got {type(value)}")
                if value.device.type != "npu":
                    raise ValueError(
                        f"argument {i} is on {value.device}, expected an NPU tensor; the tile launch ABI passes a raw device pointer"
                    )
                if not value.is_contiguous():
                    raise ValueError(f"argument {i} must be contiguous; the tile launch ABI passes a raw data pointer without strides")
                encoded.append(int(value.data_ptr()))
            elif kind in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                encoded.append(int(value))
            elif kind == "float32":
                encoded.append(struct.unpack("<i", struct.pack("<f", float(value)))[0])
            elif kind == "float64":
                encoded.append(struct.unpack("<q", struct.pack("<d", float(value)))[0])
            else:  # pragma: no cover - extract_launch_info rejects these
                raise ValueError(f"unsupported arg kind `{kind}`")
        return encoded

    def __call__(self, *args: Any) -> None:
        if len(args) != len(self.info.arg_types):
            raise ValueError(f"kernel `{self.info.name}` expects {len(self.info.arg_types)} arguments but got {len(args)}")
        arg_list = list(args)
        grid = self._compute_grid(arg_list)
        ubuf = self._compute_ubuf(arg_list)
        encoded = self._encode_args(arg_list)
        stream = _current_npu_stream()
        self._launch(self.obj_bytes, self.info.name, grid, ubuf, stream, self.info.arg_types, encoded)


@dataclasses.dataclass
class _ElfSymbol:
    """Minimal view of an ELF symbol relevant to contract checking."""

    name: str
    st_type: int  # STT_OBJECT=1 / STT_FUNC=2
    st_size: int
    st_binding: int


# ---------------------------------------------------------------------------
# ELF layout constants (generic System V ELF ABI; apply to any ELF object).
# Field offsets are fixed by the ABI per class (ELFCLASS32 / ELFCLASS64).
# ---------------------------------------------------------------------------
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32, _ELFCLASS64 = 1, 2
_ELFDATA2LSB, _ELFDATA2MSB = 1, 2
# ELF header field offsets.
_EHDR_SH_OFF_OFFSET = {32: 0x20, 64: 0x28}  # e_shoff
_EHDR_SH_ENTSIZE_OFFSET = {32: 0x2E, 64: 0x3A}  # e_shentsize
_EHDR_SH_NUM_OFFSET = {32: 0x30, 64: 0x3C}  # e_shnum
# Section header field offsets.
_SHDR_TYPE_OFFSET = 4  # sh_type (same for both classes)
_SHDR_OFFSET_OFFSET = {32: 16, 64: 24}  # sh_offset
_SHDR_SIZE_OFFSET = {32: 20, 64: 32}  # sh_size
_SHDR_LINK_OFFSET = {32: 24, 64: 40}  # sh_link (index of linked strtab)
_SHDR_ENTSIZE_OFFSET = {32: 36, 64: 56}  # sh_entsize
# Section types (SHT_*).
_SHT_SYMTAB, _SHT_STRTAB, _SHT_DYNSYM = 2, 3, 11
# Symbol entry field offsets and entry size.  Elf32_Sym and Elf64_Sym differ
# in field order: Elf32 is {st_name, st_value, st_size, st_info, st_other,
# st_shndx} while Elf64 is {st_name, st_info, st_other, st_shndx, st_value,
# st_size}, so st_info / st_size offsets must be selected per class.
_SYM_NAME_OFFSET = 0  # st_name (offset into strtab)
_SYM_INFO_OFFSET = {32: 12, 64: 4}  # st_info (low 4 bits = type, high 4 = binding)
_SYM_SIZE_OFFSET = {32: 8, 64: 16}  # st_size
_SYM_ENTRY_SIZE = {32: 16, 64: 24}
# Symbol types (STT_*).
_STT_OBJECT, _STT_FUNC = 1, 2


def _read_elf_symbols(data: bytes) -> dict[str, _ElfSymbol]:
    """
    Parse .symtab/.dynsym from a raw ELF image (zero extra dependencies).

    Verified to match pyelftools on the repo's hand-compiled .o files.
    """
    if data[:4] != _ELF_MAGIC:
        raise ValueError("invalid ELF magic (bad magic)")
    cls, endian = data[4], data[5]
    if cls not in (_ELFCLASS32, _ELFCLASS64) or endian not in (_ELFDATA2LSB, _ELFDATA2MSB):
        raise ValueError("invalid ELF class or endianness")

    # The *_OFFSET dicts are keyed by the class bit-width (32/64); translate
    # the ELFCLASS enum value (1/2) to the bit-width before indexing.
    bits = 32 if cls == _ELFCLASS32 else 64
    eo = "<" if endian == _ELFDATA2LSB else ">"
    size_t = "Q" if bits == 64 else "I"
    e_shoff = struct.unpack_from(eo + size_t, data, _EHDR_SH_OFF_OFFSET[bits])[0]
    e_shentsize, e_shnum = struct.unpack_from(eo + "HH", data, _EHDR_SH_ENTSIZE_OFFSET[bits])
    sh_offset_off = _SHDR_OFFSET_OFFSET[bits]
    sh_size_off = _SHDR_SIZE_OFFSET[bits]
    sh_link_off = _SHDR_LINK_OFFSET[bits]
    sh_entsize_off = _SHDR_ENTSIZE_OFFSET[bits]
    sym_size_off = _SYM_SIZE_OFFSET[bits]
    sym_info_off = _SYM_INFO_OFFSET[bits]
    sym_size = _SYM_ENTRY_SIZE[bits]

    # Collect the string tables (.strtab / .dynstr, SHT_STRTAB) keyed by
    # section index; .symtab/.dynsym reference their strtab via sh_link.
    strtabs: dict[int, bytes] = {}
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if struct.unpack_from(eo + "I", data, sh + _SHDR_TYPE_OFFSET)[0] != _SHT_STRTAB:
            continue
        off = struct.unpack_from(eo + size_t, data, sh + sh_offset_off)[0]
        size = struct.unpack_from(eo + size_t, data, sh + sh_size_off)[0]
        strtabs[i] = data[off : off + size]

    symbols: dict[str, _ElfSymbol] = {}
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from(eo + "I", data, sh + _SHDR_TYPE_OFFSET)[0]
        if sh_type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        off = struct.unpack_from(eo + size_t, data, sh + sh_offset_off)[0]
        size = struct.unpack_from(eo + size_t, data, sh + sh_size_off)[0]
        link = struct.unpack_from(eo + "I", data, sh + sh_link_off)[0]
        entsize = struct.unpack_from(eo + size_t, data, sh + sh_entsize_off)[0]
        sttab = strtabs.get(link, b"")
        for k in range(size // (entsize or sym_size)):
            st = off + k * (entsize or sym_size)
            st_name = struct.unpack_from(eo + "I", data, st + _SYM_NAME_OFFSET)[0]
            if st_name >= len(sttab):
                continue
            end = sttab.find(b"\x00", st_name)
            name = sttab[st_name:end].decode("utf-8", "replace") if end >= 0 else ""
            if not name:
                continue
            st_type = data[st + sym_info_off] & 0xF
            st_binding = data[st + sym_info_off] >> 4
            st_size = struct.unpack_from(eo + size_t, data, st + sym_size_off)[0]
            symbols.setdefault(name, _ElfSymbol(name, st_type, st_size, st_binding))
    return symbols


def _validate_symbol_table(info: LaunchInfo, obj_path: str | Path, obj_bytes: bytes) -> None:
    """
    Contract #1/#2 (symbol half): the .o must expose the kernel as a function.
    """
    try:
        symbols = _read_elf_symbols(obj_bytes)
    except Exception as e:
        raise ValueError(f"tile object `{obj_path}` is not a readable ELF: {e}") from e
    if not symbols:
        raise ValueError(f"tile object `{obj_path}` has no symbol table (fully stripped?); cannot verify the kernel symbol contract.")
    sym = symbols.get(info.name)
    if sym is None:
        candidates = sorted(n for n, s in symbols.items() if s.st_type == _STT_FUNC and not n.startswith("."))
        raise ValueError(
            f"tile object `{obj_path}` has no symbol `{info.name}`; "
            f"(expected kernel name = IR global_symbol). "
            f"Available function symbols: {candidates or '(none)'}"
        )
    if sym.st_type != _STT_FUNC:
        raise ValueError(
            f"symbol `{info.name}` in `{obj_path}` is not a function (type={sym.st_type}); contract #1 requires a function symbol."
        )
    if sym.st_size == 0:
        raise ValueError(f"symbol `{info.name}` in `{obj_path}` has 0 size; the kernel has no code -- wrong .o?")


def _check_dwarf_signature(info: LaunchInfo, obj_bytes: bytes) -> tuple[str, str | None]:
    """
    Best-effort parameter-signature check via DWARF (needs debug info).

    returns ("ok", None) on verified match, ("error", msg) on a definite mismatch
        and ("warn", msg) when the signature cannot be determined statically.

    e.g. non-hardwareized blockIdx -> expected+1 params ==> return error
    """
    try:
        from elftools.elf.elffile import ELFFile  # optional dependency
        import io as _io
    except Exception:
        return "warn", "pyelftools not installed; parameter-signature check skipped"
    expected = len(info.arg_types)
    try:
        elf = ELFFile(_io.BytesIO(obj_bytes))
        dwarf = elf.get_dwarf_info()
        for cu in dwarf.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_subprogram":
                    continue
                name_attr = die.attributes.get("DW_AT_name")
                if name_attr is None or name_attr.value.decode("utf-8", "replace") != info.name:
                    continue
                nparams = sum(1 for child in die.iter_children() if child.tag == "DW_TAG_formal_parameter")
                if nparams == expected:
                    return "ok", None
                hint = ""
                if nparams == expected + 1:
                    hint = (
                        " This looks like a non-hardwareized blockIdx parameter "
                        "(contract #3): the .o kernel still takes the i32 blockIdx "
                        "argument but launch only passes the handles."
                    )
                return "error", (f"kernel `{info.name}` has {nparams} formal parameters but the IR expects {expected}.{hint}")
    except Exception as e:
        # Malformed DWARF must not break binding: the whole traversal is
        # best-effort, so any parsing failure downgrades to a warning.
        return "warn", f"cannot read DWARF debug info ({e}); signature check skipped"
    return "warn", (
        f"no DWARF subprogram found for `{info.name}`; cannot statically confirm "
        "the parameter list. Make sure the .o was compiled with the blockIdx "
        "parameter hardwareized (contract #3)."
    )


def _validate_tile_obj(info: LaunchInfo, obj_path: str | Path, obj_bytes: bytes) -> None:
    """
    Contract validation for the (artifact, .o) pairing at bind time.

    Strict failures raise ValueError; undeterminable signature checks warn.
    """
    _validate_symbol_table(info, obj_path, obj_bytes)
    kind, msg = _check_dwarf_signature(info, obj_bytes)
    if kind == "error":
        raise ValueError(msg)
    if kind == "warn":
        # stacklevel=3: user code -> compile_tile_obj -> _validate_tile_obj -> warn,
        # so the warning points at the user's bind call, not tilelang internals.
        warnings.warn(f"tile object `{obj_path}`: {msg}", stacklevel=3)


def compile_tile_obj(
    func_or_artifact,
    obj_path: str | Path,
    grid_override: int | None = None,
    ubuf_override: int | None = None,
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
        Skip TIR UBUF extraction and always pass this dynamic UBUF size.
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
    obj_bytes = Path(obj_path).read_bytes()
    _validate_tile_obj(info, obj_path, obj_bytes)
    return TileObjKernel(info, obj_bytes, grid_override=grid_override, ubuf_override=ubuf_override)
