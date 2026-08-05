"""TIRX to TileLangIR MLIR translation."""

from __future__ import annotations

from typing import Any

from tvm import IRModule, tirx
from tvm.tirx import PrimFunc, PyStmtExprVisitor

from .dialect import configure_context
from ._tilelang_ops_gen import CopyOp, GemmOp, LaunchThreadOp


@tirx.functor.visitor
class TileLangIRTranslator(PyStmtExprVisitor):
    """Translate TIRX nodes into their TileLangIR MLIR representation.

    This initial storage-focused implementation intentionally flattens TIRX
    control and scope nodes into the ``func.func`` entry block.  It preserves
    the values needed by ``tl.tileop.vadd`` while Kernel, For, SBlock, and copy
    conversion are introduced separately.
    """

    def __init__(self) -> None:
        super().__init__()
        self.value_map: dict[object, Any] = {} # value_map stores mappings from TIRX objects to MLIR Values, ensuring that the same TIRX object maps to the same MLIR Value
        self._arith: Any = None
        self._ir: Any = None
        self._func: Any = None
        self._linalg: Any = None
        self._memref: Any = None

    def translate(self, source_module: IRModule) -> str:
        # 1. Check that the input is a tvm.IRModule
        if not isinstance(source_module, IRModule):
            raise TypeError(
                "TileLangIR codegen expects tvm.IRModule, "
                f"but received {type(source_module).__name__}."
            )
        # 2. Load and validate the MLIR Python bindings
        try:
            from mlir import ir
            from mlir.dialects import arith, func, linalg, memref
        except ImportError as exc:
            raise ImportError(
                "TileLangIR codegen requires LLVM's official MLIR Python bindings. "
                "Install them with `python -m pip install mlir-python-bindings "
                "-f https://llvm.github.io/eudsl`."
            ) from exc

        self._arith = arith
        self._ir = ir
        self._func = func
        self._linalg = linalg
        self._memref = memref

        with ir.Context() as context:
            try:
                configure_context(context)
                with ir.Location.unknown(context):
                    # 3. Create an empty MLIR module
                    module = ir.Module.create()
                    with ir.InsertionPoint(module.body):
                        # 4. Iterate over the PrimFuncs
                        for global_var, base_func in source_module.functions.items():
                            if not isinstance(base_func, PrimFunc):
                                raise TypeError(
                                    "TileLangIR codegen currently accepts only TIRX PrimFunc "
                                    f"entries, but @{global_var.name_hint} is "
                                    f"{type(base_func).__name__}."
                                )
                            self._emit_prim_func(global_var.name_hint, base_func)
                    module.operation.verify()
                    source = str(module)
            finally:
                self.value_map.clear()
                self._arith = None
                self._ir = None
                self._func = None
                self._linalg = None
                self._memref = None

        return source

    def _emit_prim_func(self, name: str, prim_func: PrimFunc) -> None:
        """Emit one function and materialize its TIRX Buffer parameters."""

        self.value_map.clear()
        # 1. Parse the function parameters and create the MLIR function
        input_types = [self._parameter_type(param, prim_func) for param in prim_func.params]
        function_type = self._ir.FunctionType.get(input_types, [])
        function_op = self._func.FuncOp(name, function_type)
        entry_block = function_op.add_entry_block()

        # 2. Set the MLIR insertion point inside func.func
        with self._ir.InsertionPoint(entry_block):
            # 3. Process the function parameters
            for param, block_argument in zip(prim_func.params, entry_block.arguments):
                self._insert_value(param, block_argument)
                if param in prim_func.buffer_map:
                    buffer = prim_func.buffer_map[param]
                    self._insert_value(buffer.data, block_argument)

            for param in prim_func.params:
                if param in prim_func.buffer_map:
                    self._emit_parameter_buffer(prim_func.buffer_map[param])

            # 4. Traverse the TIRX tree
            self.visit_stmt(prim_func.body)
            self._func.ReturnOp([])
    #############
    # Function parameter handling
    #############
    def _parameter_type(self, param: Any, prim_func: PrimFunc) -> Any:
        if param not in prim_func.buffer_map:
            return self._dtype_type(param.dtype)

        buffer = prim_func.buffer_map[param]
        dynamic = self._ir.ShapedType.get_dynamic_size()
        return self._ir.MemRefType.get(
            [dynamic],
            self._dtype_type(buffer.dtype),
            memory_space=self._memory_scope(buffer),
        )

    def _emit_parameter_buffer(self, buffer: Any) -> Any:
        backing = self._get_value(buffer.data)
        shape = self._static_shape(buffer)
        strides = self._buffer_strides(buffer, shape)
        offset = self._static_int(buffer.elem_offset, f"{buffer.name}.elem_offset")
        result_type = self._buffer_memref_type(buffer, shape, strides, offset)
        view = self._memref.ReinterpretCastOp(
            result_type,
            backing,
            [],
            [],
            [],
            [offset],
            shape,
            strides,
        ).result
        self._insert_value(buffer, view)
        return view

    def _buffer_memref_type(
        self,
        buffer: Any,
        shape: list[int],
        strides: list[int],
        offset: int,
    ) -> Any:
        layout = self._ir.StridedLayoutAttr.get(offset, strides)
        return self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            layout=layout,
            memory_space=self._memory_scope(buffer),
        )
    #############
    # TIRX tree traversal helper functions
    #############
    def _emit_alloc_buffer(self, buffer: Any) -> Any:
        scope = str(buffer.scope())
        if scope in ("shared", "shared.dyn"):
            return self._emit_shared_buffer(buffer)

        # Register fragments are logical typed allocations.  Unlike shared
        # memory, they do not use a reusable byte-level backing allocation.
        shape = self._static_shape(buffer)
        result_type = self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            memory_space=self._memory_scope(buffer),
        )
        allocated = self._memref.AllocOp(result_type, [], []).result
        self._insert_value(buffer.data, allocated)
        self._insert_value(buffer, allocated)
        return allocated

    def _emit_shared_buffer(self, buffer: Any) -> Any:
        """Emit shared storage as a byte allocation plus a typed view."""

        shape = self._static_shape(buffer)
        byte_size = self._num_elements(shape) * self._dtype_nbytes(buffer.dtype)
        memory_space = self._memory_scope(buffer)
        raw_type = self._ir.MemRefType.get(
            [byte_size],
            self._ir.IntegerType.get_signless(8),
            memory_space=memory_space,
        )
        raw = self._memref.AllocOp(raw_type, [], []).result
        self._insert_value(buffer.data, raw)

        typed_type = self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            memory_space=memory_space,
        )
        index_type = self._ir.IndexType.get()
        byte_shift = self._arith.ConstantOp(
            index_type, self._ir.IntegerAttr.get(index_type, 0)
        ).result
        typed = self._memref.ViewOp(typed_type, raw, byte_shift, []).result
        self._insert_value(buffer, typed)
        return typed

    def _emit_region(self, call: tirx.Call) -> Any:
        if len(call.args) < 3:
            raise ValueError("tl.tileop.region expects BufferLoad, access type, and extents")

        buffer_load = call.args[0]
        if not isinstance(buffer_load, tirx.BufferLoad):
            raise TypeError(
                "tl.tileop.region expects a BufferLoad as arg0, "
                f"but received {type(buffer_load).__name__}."
            )

        buffer = buffer_load.buffer
        backing = self._get_value(buffer)
        buffer_shape = self._static_shape(buffer)
        buffer_strides = self._buffer_strides(buffer, buffer_shape)
        indices = [
            self._static_int(index, f"{buffer.name} region index") for index in buffer_load.indices
        ]
        if len(indices) != len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(indices)} indices for rank-{len(buffer_strides)} Buffer."
            )

        offset = self._static_int(buffer.elem_offset, f"{buffer.name}.elem_offset")
        offset += sum(index * stride for index, stride in zip(indices, buffer_strides))
        sizes = [self._static_int(extent, f"{buffer.name} region extent") for extent in call.args[2:]]
        if len(sizes) > len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(sizes)} extents for rank-{len(buffer_strides)} Buffer."
            )
        strides = buffer_strides[-len(sizes) :]
        layout = self._ir.StridedLayoutAttr.get(offset, strides)
        result_type = self._ir.MemRefType.get(
            sizes,
            self._dtype_type(buffer.dtype),
            layout=layout,
            memory_space=self._memory_scope(buffer),
        )
        view_op = self._memref.ReinterpretCastOp(
            result_type,
            backing,
            [],
            [],
            [],
            [offset],
            sizes,
            strides,
        )
        self._insert_value(call, view_op.result)
        return view_op.result

    def _emit_vadd(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vadd expects 3 arguments, but received {len(call.args)}")
        for operand in call.args:
            self.visit_expr(operand)

        src0 = self._get_value(call.args[0])
        src1 = self._get_value(call.args[1])
        dst = self._get_value(call.args[2])
        self._linalg.add(src0, src1, outs=[dst])

    def _memory_scope(self, buffer: Any) -> Any:
        scope = str(buffer.scope())
        if scope in ("", "global"):
            address_space = 0
        elif scope in ("shared", "shared.dyn"):
            address_space = 1
        elif scope == "local.fragment":
            address_space = 2
        else:
            raise NotImplementedError(
                f"TileLangIR has no memory-space mapping for TIRX scope {scope!r}."
            )
        return self._ir.IntegerAttr.get(self._ir.IntegerType.get_signless(64), address_space)

    @staticmethod
    def _num_elements(shape: list[int]) -> int:
        count = 1
        for extent in shape:
            count *= extent
        return count

    @staticmethod
    def _dtype_nbytes(dtype: Any) -> int:
        bits = int(dtype.bits) * int(dtype.lanes)
        return max(1, (bits + 7) // 8)

    def _dtype_type(self, dtype: Any) -> Any:
        dtype_name = str(dtype)
        if dtype_name == "bool":
            return self._ir.IntegerType.get_signless(1)
        if dtype_name == "bfloat16":
            return self._ir.BF16Type.get()
        if dtype_name == "float16":
            return self._ir.F16Type.get()
        if dtype_name == "float32":
            return self._ir.F32Type.get()
        if dtype_name == "float64":
            return self._ir.F64Type.get()
        if dtype_name.startswith(("int", "uint")):
            prefix_length = 3 if dtype_name.startswith("int") else 4
            return self._ir.IntegerType.get_signless(int(dtype_name[prefix_length:]))
        raise TypeError(f"Unsupported TIRX dtype in TileLangIR codegen: {dtype_name}.")

    def _static_shape(self, buffer: Any) -> list[int]:
        return [self._static_int(extent, f"{buffer.name} shape") for extent in buffer.shape]

    def _buffer_strides(self, buffer: Any, shape: list[int]) -> list[int]:
        if buffer.strides:
            return [self._static_int(stride, f"{buffer.name} stride") for stride in buffer.strides]

        strides = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            strides[index] = strides[index + 1] * shape[index + 1]
        return strides

    @staticmethod
    def _static_int(expr: Any, description: str) -> int:
        if isinstance(expr, tirx.IntImm):
            return int(expr.value)
        raise NotImplementedError(
            f"Current TileLangIR storage lowering requires static {description}, but got {expr}."
        )

    @staticmethod
    def _call_op_name(call: tirx.Call) -> str:
        return call.op.name

    #############
    # value_map helper methods
    #############
    # Insert a mapping from a TIRX object into value_map
    def _insert_value(self, tirx_object: object, mlir_value: Any) -> None:
        if tirx_object in self.value_map:
            if self.value_map[tirx_object] == mlir_value:
                return
            raise ValueError(f"TIRX value is already present in value_map: {tirx_object}")
        self.value_map[tirx_object] = mlir_value

    # Get the MLIR Value mapped to a TIRX object
    def _get_value(self, tirx_object: object) -> Any:
        try:
            return self.value_map[tirx_object]
        except KeyError as exc:
            raise KeyError(f"TIRX value has not been lowered yet: {tirx_object}") from exc

    #############
    # Overridden visitor methods
    #############
    # Handle different TIRX node types; the override method names are fixed
    # The following methods are dispatched by visit_stmt.
    def visit_sblock_realize_(self, op: tirx.SBlockRealize) -> None:
        self.visit_stmt(op.block)

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        for buffer in op.alloc_buffers:
            self._emit_alloc_buffer(buffer)
        self.visit_stmt(op.body)

    def visit_for_(self, op: tirx.For) -> None:
        self.visit_stmt(op.body)

    def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
        self.visit_stmt(op.body)

    # The following methods are dispatched by visit_expr.
    # Operations registered under src/op are represented as Call nodes.
    def visit_call_(self, op: tirx.Call) -> None:
        op_name = self._call_op_name(op)
        if op_name == "tl.tileop.region":
            self._emit_region(op)
        elif op_name == "tl.tileop.vadd":
            self._emit_vadd(op)
