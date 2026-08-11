"""TIRX to TileLangIR MLIR translation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from tvm import IRModule, tirx
from tvm.tirx import PrimFunc, PyStmtExprVisitor

from .dialect import configure_context
from ._tilelang_ops_gen import CopyOp, GemmOp, ScopeOp


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
        self.value_map: dict[int, Any] = {} # value_map stores mappings from TIRX object identities to MLIR Values, ensuring that the same TIRX object maps to the same MLIR Value
        self._arith: Any = None
        self._ir: Any = None
        self._func: Any = None
        self._linalg: Any = None
        self._math: Any = None
        self._memref: Any = None
        self._scf: Any = None

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
            from mlir.dialects import arith, func, linalg, math, memref, scf
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
        self._math = math
        self._memref = memref
        self._scf = scf

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
                self._math = None
                self._memref = None
                self._scf = None

        return source

    def _emit_prim_func(self, name: str, prim_func: PrimFunc) -> None:
        """Emit one function and materialize its TIRX Buffer parameters."""

        self.value_map.clear()
        block_idx_bindings = self._collect_block_idx_bindings(prim_func.body)

        # 1. Parse the function parameters and create the MLIR function
        input_types = [self._parameter_type(param, prim_func) for param in prim_func.params]
        arg_base = len(input_types)
        i32_type = self._ir.IntegerType.get_signless(32)
        input_types.extend(i32_type for _ in block_idx_bindings)
        function_type = self._ir.FunctionType.get(input_types, [])
        function_op = self._func.FuncOp(name, function_type)
        self._set_block_idx_attr(function_op, block_idx_bindings, arg_base)
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

            for offset, (var, _extent, _tag) in enumerate(block_idx_bindings):
                self._insert_value(var, entry_block.arguments[arg_base + offset])

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

    @staticmethod
    def _collect_block_idx_bindings(body: Any) -> list[tuple[Any, Any, str]]:
        bindings: list[tuple[Any, Any, str]] = []
        seen: set[int] = set()

        @tirx.functor.visitor
        class Collector(PyStmtExprVisitor):
            def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
                if str(op.attr_key) == "thread_extent":
                    node = op.node
                    tag = str(getattr(node, "thread_tag", "") or "")
                    var = getattr(node, "var", None)
                    if tag.startswith("blockIdx.") and var is not None:
                        key = id(var)
                        if key not in seen:
                            seen.add(key)
                            bindings.append((var, op.value, tag))
                self.visit_stmt(op.body)

        Collector().visit_stmt(body)
        order = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
        bindings.sort(key=lambda item: order.get(item[2], 99))
        return bindings

    def _set_block_idx_attr(
        self,
        function_op: Any,
        block_idx_bindings: list[tuple[Any, Any, str]],
        arg_base: int,
    ) -> None:
        """Record which `func.func` argument carries `blockIdx.x`.

        Downstream reads the block index from a hardware instruction rather than
        from the launch grid, so the attribute is a parameter position, not the
        grid extent.
        """
        for offset, (_var, _extent, tag) in enumerate(block_idx_bindings):
            if tag != "blockIdx.x":
                continue
            i64 = self._ir.IntegerType.get_signless(64)
            function_op.attributes["BlockIdx"] = self._ir.IntegerAttr.get(
                i64, arg_base + offset
            )
            return

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

    def _emit_linear_offset(
        self,
        buffer: Any,
        indices: list[tirx.PrimExpr],
        buffer_strides: list[int],
    ) -> Any:
        if isinstance(buffer.elem_offset, tirx.IntImm) and all(
            isinstance(index, tirx.IntImm) for index in indices
        ):
            offset = int(buffer.elem_offset.value)
            offset += sum(
                int(index.value) * stride for index, stride in zip(indices, buffer_strides)
            )
            return offset

        index_type = self._ir.IndexType.get()
        offset = self._cast_to_index(self._get_or_create_expr_value(buffer.elem_offset))
        for index, stride in zip(indices, buffer_strides):
            index_value = self._cast_to_index(self._get_or_create_expr_value(index))
            stride_value = self._arith.ConstantOp(
                index_type, self._ir.IntegerAttr.get(index_type, stride)
            ).result
            scaled_index = self._arith.MulIOp(index_value, stride_value).result
            offset = self._arith.AddIOp(offset, scaled_index).result
        return offset

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
        indices = list(buffer_load.indices)
        if len(indices) != len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(indices)} indices for rank-{len(buffer_strides)} Buffer."
            )

        raw_extents = list(call.args[2:])
        if len(raw_extents) > len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(raw_extents)} extents for rank-{len(buffer_strides)} Buffer."
            )
        region_strides = buffer_strides[-len(raw_extents) :]
        sizes, strides = self._squeeze_unit_dims(raw_extents, region_strides)

        offset = self._emit_linear_offset(buffer, indices, buffer_strides)
        dynamic = self._ir.ShapedType.get_dynamic_size()
        if isinstance(offset, int):
            dynamic_offsets = []
            static_offsets = [offset]
            layout_offset = offset
        else:
            dynamic_offsets = [offset]
            static_offsets = [dynamic]
            layout_offset = dynamic

        static_sizes: list[int] = []
        size_operands: list[Any] = []
        for size in sizes:
            if isinstance(size, int):
                static_sizes.append(size)
            else:
                static_sizes.append(dynamic)
                size_operands.append(
                    self._cast_to_index(self._get_or_create_expr_value(size))
                )

        layout = self._ir.StridedLayoutAttr.get(layout_offset, strides)
        result_type = self._ir.MemRefType.get(
            static_sizes,
            self._dtype_type(buffer.dtype),
            layout=layout,
            memory_space=self._memory_scope(buffer),
        )
        view_op = self._memref.ReinterpretCastOp(
            result_type,
            backing,
            dynamic_offsets,
            size_operands,
            [],
            static_offsets,
            static_sizes,
            strides,
        )
        self._insert_value(call, view_op.result)
        return view_op.result

    @staticmethod
    def _squeeze_unit_dims(
        extents: list[Any],
        strides: list[int],
    ) -> tuple[list[Any], list[int]]:
        sizes: list[Any] = []
        kept_strides: list[int] = []
        for extent, stride in zip(extents, strides):
            static = TileLangIRTranslator._try_static_int(extent)
            if static == 1:
                continue
            sizes.append(static if static is not None else extent)
            kept_strides.append(stride)
        return sizes, kept_strides

    def _emit_vadd(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vadd expects 3 arguments, but received {len(call.args)}")
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        self._linalg.add(src0, src1, outs=[dst])

    def _emit_vexpdif(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(
                f"tl.tileop.vexpdif expects 3 arguments, but received {len(call.args)}"
            )
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])

        dst_type = dst.type
        rank = dst_type.rank
        element_type = dst_type.element_type

        indexing_map = self._ir.AffineMap.get(
            rank, 0, [self._ir.AffineExpr.get_dim(index) for index in range(rank)]
        )
        indexing_maps = self._ir.ArrayAttr.get(
            [self._ir.AffineMapAttr.get(indexing_map)] * 3
        )
        iterator_types = self._ir.ArrayAttr.get(
            [self._ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        )
        generic = self._linalg.GenericOp(
            result_tensors=[],
            inputs=[src0, src1],
            outputs=[dst],
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
        )
        block = generic.regions[0].blocks.append(element_type, element_type, element_type)
        with self._ir.InsertionPoint(block):
            diff = self._arith.SubFOp(block.arguments[0], block.arguments[1]).result
            result = self._math.ExpOp(diff).result
            self._linalg.YieldOp([result])

    def _emit_copy(self, call: tirx.Call) -> None:
        if len(call.args) < 2:
            raise ValueError(
                f"tl.tileop.copy expects at least 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        ann = self._call_annotations(call)
        CopyOp(
            src,
            dst,
            split_dim=self._optional_int(ann.get("split_dim")),
            transpose=True if self._as_bool(ann.get("transpose", False)) else None,
        )

    def _emit_gemm(self, call: tirx.Call) -> None:
        # T.gemm Call arg layout (see tilelang/language/gemm_op.py):
        #   0: A region, 1: B region, 2: C region,
        #   3: transpose_A (bool), 4: transpose_B (bool),
        #   5: M, 6: N, 7: K, 8: policy,
        #   9: clear_accum (bool), 10-18: strides/offsets/etc (unused by GemmOp)
        if len(call.args) < 10:
            raise ValueError(
                f"tl.tileop.gemm expects at least 10 arguments, but received {len(call.args)}"
            )
        a = self._get_or_create_expr_value(call.args[0])
        b = self._get_or_create_expr_value(call.args[1])
        c = self._get_or_create_expr_value(call.args[2])
        transpose_a = self._as_bool(call.args[3])
        transpose_b = self._as_bool(call.args[4])
        clear_accum = self._as_bool(call.args[9])
        GemmOp(
            a,
            b,
            c,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            clear_accum=clear_accum,
        )

    def _cast_to_index(self, value: Any) -> Any:
        index_type = self._ir.IndexType.get()
        if value.type == index_type:
            return value
        if isinstance(value.type, self._ir.IntegerType):
            return self._arith.IndexCastOp(index_type, value).result
        raise TypeError(f"Cannot cast MLIR value of type {value.type} to index.")

    def _unify_integer_operands(self, lhs: Any, rhs: Any) -> tuple[Any, Any]:
        if lhs.type == rhs.type:
            return lhs, rhs
        index_type = self._ir.IndexType.get()
        if lhs.type == index_type or rhs.type == index_type:
            return self._cast_to_index(lhs), self._cast_to_index(rhs)
        raise TypeError(
            f"Cannot unify MLIR integer operand types {lhs.type} and {rhs.type}."
        )

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
        value = TileLangIRTranslator._try_static_int(expr)
        if value is not None:
            return value
        raise NotImplementedError(
            f"Current TileLangIR storage lowering requires static {description}, but got {expr}."
        )

    @staticmethod
    def _try_static_int(expr: Any) -> int | None:
        if isinstance(expr, tirx.IntImm):
            return int(expr.value)
        if isinstance(expr, int):
            return expr
        return None

    @staticmethod
    def _call_op_name(call: tirx.Call) -> str:
        return call.op.name

    #############
    # value_map helper methods
    #############
    # Insert a mapping from a TIRX object into value_map
    def _insert_value(self, tirx_object: object, mlir_value: Any) -> None:
        key = tirx_object.id_
        if key in self.value_map:
            if self.value_map[key] == mlir_value:
                return
            raise ValueError(f"TIRX value is already present in value_map: {tirx_object}")
        self.value_map[key] = mlir_value

    # Get the MLIR Value mapped to a TIRX object
    def _get_value(self, tirx_object: object) -> Any:
        key = tirx_object.id_
        try:
            return self.value_map[key]
        except KeyError as exc:
            raise KeyError(f"TIRX value has not been lowered yet: {tirx_object}") from exc

    def _get_or_create_expr_value(self, expr: tirx.PrimExpr) -> Any:
        """Get or emit the MLIR Value represented by a TIRX PrimExpr."""

        key = expr.id_
        if key in self.value_map:
            return self._get_value(expr)

        self.visit_expr(expr)

        try:
            return self._get_value(expr)
        except KeyError as exc:
            raise NotImplementedError(
                "TIRX expression lowering did not produce an MLIR Value for "
                f"{type(expr).__name__}: {expr}"
            ) from exc

    @contextmanager
    def _scoped_value_map(self) -> Generator[None, None, None]:
        """Lexically scope the value_map to the enclosing ``with`` block.

        Values created inside a child region (e.g. a ``tilelang.scope`` or
        ``scf.for`` body) must not be referenced by the enclosing block,
        otherwise an SSA value defined in the child region could be used where
        it does not dominate the use site. Save the current key set on entry
        and drop every newly-added key on exit, even if an exception
        propagates.
        """
        saved_keys = set(self.value_map)
        try:
            yield
        finally:
            for key in list(self.value_map):
                if key not in saved_keys:
                    del self.value_map[key]




    #############
    # Overridden visitor methods
    #############
    # Handle different TIRX node types; the override method names are fixed
    # The following methods are dispatched by visit_stmt.
    def visit_sblock_realize_(self, op: tirx.SBlockRealize) -> None:
        self.visit_stmt(op.block)

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        name = str(getattr(op, "name_hint", "") or "")
        if name == "SimdVF":
            self._emit_simdvf_scope(op)
            return

        for buffer in op.alloc_buffers:
            self._emit_alloc_buffer(buffer)
        self.visit_stmt(op.body)

    def _emit_simdvf_scope(self, op: tirx.SBlock) -> None:
        """Emit a ``tilelang.scope`` region for a ``T.SimdVF()`` SBlock.

        The SBlock's alloc_buffers and body live in the scope's nested region.
        Values created inside are scoped to that region and dropped from the
        value_map afterwards, mirroring visit_for_: an SSA value defined in a
        child region must not be referenced by the enclosing block.
        """
        scope_op = ScopeOp(simd_attr="simd")
        body_block = self._ir.Block.create_at_start(scope_op.body)
        with self._ir.InsertionPoint(body_block), self._scoped_value_map():
            for buffer in op.alloc_buffers:
                self._emit_alloc_buffer(buffer)
            self.visit_stmt(op.body)

    def visit_for_(self, op: tirx.For) -> None:
        index_type = self._ir.IndexType.get()
        lower = self._cast_to_index(self._get_or_create_expr_value(op.min))
        extent = self._cast_to_index(self._get_or_create_expr_value(op.extent))
        upper = self._arith.AddIOp(lower, extent).result
        if op.step is not None:
            step = self._cast_to_index(self._get_or_create_expr_value(op.step))
        else:
            step = self._arith.ConstantOp(
                index_type, self._ir.IntegerAttr.get(index_type, 1)
            ).result

        for_op = self._scf.ForOp(lower, upper, step)
        loop_kind = self._loop_kind_attr(op)
        if loop_kind is not None:
            for_op.attributes["tilelang.loop_kind"] = loop_kind
        self._copy_loop_annotations(op, for_op)

        # Scope the value_map to this region. TIRX may alias the same node (e.g.
        # an interned IntImm loop bound) at several tree positions. Values created
        # inside the scf.for body must not be reused outside it, otherwise an
        # arith.index_cast emitted in a parent region could reference an operand
        # defined in this (child) region, violating SSA dominance.
        body_keys = set(self.value_map)
        self._insert_value(op.loop_var, for_op.induction_variable)
        with self._ir.InsertionPoint(for_op.body):
            self.visit_stmt(op.body)
            self._scf.YieldOp([])
        for key in list(self.value_map):
            if key not in body_keys:
                del self.value_map[key]

    def _loop_kind_attr(self, op: tirx.For) -> Any:
        """Classify a TIRX For into a ``tilelang.loop_kind`` string attribute.

        Pipelined loops are represented in TIRX as Serial loops carrying
        ``num_stages``/``tl_pipeline_*`` annotations (see README.md).
        """
        kind = op.kind
        if kind == tirx.ForKind.SERIAL:
            if op.annotations and "num_stages" in op.annotations:
                return self._ir.StringAttr.get("pipelined")
            return self._ir.StringAttr.get("serial")
        if kind == tirx.ForKind.PARALLEL:
            return self._ir.StringAttr.get("parallel")
        if kind == tirx.ForKind.VECTORIZED:
            return self._ir.StringAttr.get("vectorized")
        if kind == tirx.ForKind.UNROLLED:
            return self._ir.StringAttr.get("unrolled")
        # THREAD_BINDING is handled by tilelang.launch_thread, not scf.for.
        return None

    _PIPELINE_ANNOTATION_KEYS = (
        "num_stages",
        "tl_pipeline_order",
        "tl_pipeline_stage",
        "tl_pipeline_group",
    )

    def _copy_loop_annotations(self, op: tirx.For, for_op: Any) -> None:
        """Mirror pipeline and loop annotations onto the scf.for operation."""
        if not op.annotations:
            return
        for key in self._PIPELINE_ANNOTATION_KEYS:
            if key not in op.annotations:
                continue
            try:
                for_op.attributes[key] = self._annotation_value_to_attr(op.annotations[key])
            except NotImplementedError:
                continue

    def _annotation_value_to_attr(self, value: Any) -> Any:
        if isinstance(value, tirx.IntImm):
            return self._ir.IntegerAttr.get(
                self._ir.IntegerType.get_signless(64), int(value.value)
            )
        if isinstance(value, tirx.FloatImm):
            return self._ir.FloatAttr.get(self._ir.F64Type.get(), float(value.value))
        # tirx.Array and Python list/tuple both iterate element-wise.
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            elements = [self._annotation_value_to_attr(v) for v in value]
            return self._ir.ArrayAttr.get(elements)
        raise NotImplementedError(
            f"TileLangIR loop annotation lowering does not support {type(value).__name__}: {value}"
        )

    def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
        self.visit_stmt(op.body)

    # The following methods are dispatched by visit_expr.
    def visit_var_(self, op: tirx.Var) -> None:
        self._get_value(op)

    def visit_int_imm_(self, op: tirx.IntImm) -> None:
        if op.id_ in self.value_map:
            return
        result_type = self._dtype_type(op.dtype)
        value = self._arith.ConstantOp(
            result_type,
            self._ir.IntegerAttr.get(result_type, int(op.value)),
        ).result
        self._insert_value(op, value)

    def visit_float_imm_(self, op: tirx.FloatImm) -> None:
        if op.id_ in self.value_map:
            return
        result_type = self._dtype_type(op.dtype)
        value = self._arith.ConstantOp(
            result_type,
            self._ir.FloatAttr.get(result_type, float(op.value)),
        ).result
        self._insert_value(op, value)

    def visit_buffer_load_(self, op: tirx.BufferLoad) -> None:
        """
        Lower TIRX BufferLoad:

            buffer[i, j]

        into:

            memref.load %buffer[%i, %j]
        """

        buffer = op.buffer

        memref_value = self._get_value(buffer)

        indices = [
            self._cast_to_index(
                self._get_or_create_expr_value(index)
            )
            for index in op.indices
        ]

        loaded = self._memref.LoadOp(
            memref_value,
            indices,
        ).result

        self._insert_value(op, loaded)


    def visit_buffer_store_(self, op: tirx.BufferStore) -> None:
        """
        Lower TIRX BufferStore:

            buffer[i, j] = value

        into:

            memref.store value, buffer[%i, %j]
        """

        buffer = op.buffer

        memref_value = self._get_value(buffer)

        value = self._get_or_create_expr_value(op.value)

        indices = [
            self._cast_to_index(
                self._get_or_create_expr_value(index)
            )
            for index in op.indices
        ]

        self._memref.StoreOp(
            value,
            memref_value,
            indices,
        )

    def _arith_binary_op(
        self, dtype: Any, lhs: Any, rhs: Any, integer_op: str, float_op: str
    ) -> Any:
        """Emit an arith binary op, selecting the integer or float variant by the TIRX dtype."""
        is_float = isinstance(self._dtype_type(dtype), self._ir.FloatType)
        if not is_float:
            lhs, rhs = self._unify_integer_operands(lhs, rhs)
        name = float_op if is_float else integer_op
        return getattr(self._arith, name)(lhs, rhs).result

    def visit_add_(self, op: tirx.Add) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "AddIOp", "AddFOp"))

    def visit_sub_(self, op: tirx.Sub) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "SubIOp", "SubFOp"))

    def visit_mul_(self, op: tirx.Mul) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "MulIOp", "MulFOp"))

    def visit_div_(self, op: tirx.Div) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "DivSIOp", "DivFOp"))

    # Add all handlers for native TIRX expressions here.

    # Operations registered under src/op are represented as Call nodes.
    def visit_call_(self, op: tirx.Call) -> None:
        op_name = self._call_op_name(op)
        if op_name == "tl.tileop.region":
            self._emit_region(op)
        elif op_name == "tl.tileop.vadd":
            self._emit_vadd(op)
        elif op_name == "tl.tileop.vexpdif":
            self._emit_vexpdif(op)
        elif op_name == "tl.tileop.copy":
            self._emit_copy(op)
        elif op_name == "tl.tileop.gemm":
            self._emit_gemm(op)

    @staticmethod
    def _call_annotations(call: tirx.Call) -> dict[str, Any]:
        ann = getattr(call, "annotations", None)
        if ann is None:
            attrs = getattr(call, "attrs", None)
            if attrs is None:
                return {}
            try:
                return dict(attrs)
            except TypeError:
                return {}
        try:
            return dict(ann)
        except TypeError:
            return {}

    @staticmethod
    def _as_bool(expr: Any, default: bool = False) -> bool:
        if isinstance(expr, tirx.IntImm):
            return bool(expr.value)
        if isinstance(expr, bool):
            return expr
        return default

    @staticmethod
    def _optional_int(expr: Any) -> int | None:
        if expr is None:
            return None
        if isinstance(expr, tirx.IntImm):
            return int(expr.value)
        if isinstance(expr, int):
            return expr
        return None
