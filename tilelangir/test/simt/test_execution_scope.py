"""Language and TileLangIR contracts for SIMD/SIMT execution scopes."""

from __future__ import annotations

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _mixed_discrete_gather():
    @T.prim_func
    def main(
        src: T.Tensor((4096,), T.float32),
        indices: T.Tensor((128,), T.int32),
        out: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            with T.SimdVF():
                T.evaluate(0)

            with T.SimtVF(threads=1024):
                for i in T.Parallel(128):
                    out[i] = src[indices[i]]

    return main


def _simt_copy_with_multiple_items_per_thread():
    @T.prim_func
    def main(
        src: T.Tensor((128,), T.float32),
        out: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            with T.SimtVF(threads=32):
                for i in T.Parallel(128):
                    out[i] = src[i]

    return main


def _simt_two_dimensional_logical_threads():
    @T.prim_func
    def main(
        src: T.Tensor((8, 16), T.float32),
        out: T.Tensor((8, 16), T.float32),
    ):
        with T.Kernel(1, threads=1):
            with T.SimtVF(threads=32):
                for x, y in T.Parallel(8, 16):
                    out[x, y] = src[x, y]

    return main


def _simt_four_dimensional_logical_threads():
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            with T.SimtVF(threads=32):
                for x, y, z, w in T.Parallel(1, 1, 1, 1):
                    out[x + y + z + w] = 0

    return main


def _simt_scope(threads=None):
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            scope = T.SimtVF() if threads is None else T.SimtVF(threads)
            with scope:
                out[0] = 0

    return main


def _collect_nodes(root, node_type):
    nodes = []

    def visitor(node):
        if isinstance(node, node_type):
            nodes.append(node)

    tvm.tirx.stmt_functor.post_order_visit(root, visitor)
    return nodes


@tilelang.testing.requires_package("mlir")
def test_mixed_scope_contract_across_tirx_and_tilelang_ir():
    func = _mixed_discrete_gather()
    blocks = {
        block.name_hint: block
        for block in _collect_nodes(func.body, tvm.tirx.SBlock)
    }
    simd_scope = blocks["SimdVF"]
    simt_scope = blocks["SimtVF"]

    assert str(simd_scope.annotations["tl.execution_scope"]) == "simd"
    assert str(simt_scope.annotations["tl.execution_scope"]) == "simt"
    assert int(simt_scope.annotations["tl.simt_threads"]) == 1024

    parallel_loops = [
        loop
        for loop in _collect_nodes(simt_scope.body, tvm.tirx.For)
        if loop.kind == tvm.tirx.ForKind.PARALLEL
    ]
    assert len(parallel_loops) == 1
    assert int(parallel_loops[0].extent) == 128
    assert "tl.parallel_rank" not in parallel_loops[0].annotations
    assert {
        load.buffer.name
        for load in _collect_nodes(simt_scope.body, tvm.tirx.BufferLoad)
    } == {"src", "indices"}
    assert [
        store.buffer.name
        for store in _collect_nodes(simt_scope.body, tvm.tirx.BufferStore)
    ] == ["out"]

    source = tilelang.lower(func, target="tile").kernel_source
    assert source.count('"tilelang.scope"') == 2
    assert "tilelang.scope_yield" not in source
    assert "mode = #tilelang.scope_mode<simd>" in source
    assert "mode = #tilelang.scope_mode<simt>" in source
    assert "threads = 1024 : ui32" in source
    assert "arith.constant 128" in source
    assert "thread_extents" not in source
    assert source.count("memref.load") == 2
    assert "memref.store" in source
    assert "tilelang.gather" not in source
    assert source.count("scf.forall") == 1
    assert 'tilelang.logical_thread_axes = ["x"]' in source
    assert 'tilelang.loop_kind = "parallel"' in source


@tilelang.testing.requires_package("mlir")
def test_multidimensional_parallel_contract_across_tirx_and_tilelang_ir():
    func = _simt_two_dimensional_logical_threads()
    blocks = {
        block.name_hint: block
        for block in _collect_nodes(func.body, tvm.tirx.SBlock)
    }
    loops = [
        loop
        for loop in _collect_nodes(blocks["SimtVF"].body, tvm.tirx.For)
        if loop.kind == tvm.tirx.ForKind.PARALLEL
    ]
    assert len(loops) == 2
    assert all("tl.parallel_rank" not in loop.annotations for loop in loops)

    source = tilelang.lower(func, target="tile").kernel_source
    assert "threads = 32 : ui32" in source
    assert source.count("scf.forall") == 1
    assert 'tilelang.logical_thread_axes = ["x", "y"]' in source
    assert source.count("scf.for ") == 0


@tilelang.testing.requires_package("mlir")
def test_default_simt_threads_across_tirx_and_tilelang_ir():
    func = _simt_scope()
    blocks = {
        block.name_hint: block
        for block in _collect_nodes(func.body, tvm.tirx.SBlock)
    }
    assert int(blocks["SimtVF"].annotations["tl.simt_threads"]) == 1024

    source = tilelang.lower(func, target="tile").kernel_source
    assert "threads = 1024 : ui32" in source


@tilelang.testing.requires_package("mlir")
def test_physical_threads_are_separate_from_logical_work_items():
    source = tilelang.lower(
        _simt_copy_with_multiple_items_per_thread(), target="tile"
    ).kernel_source

    assert "threads = 32 : ui32" in source
    assert "arith.constant 128" in source
    assert source.count("scf.forall") == 1
    assert 'tilelang.logical_thread_axes = ["x"]' in source
    assert 'tilelang.loop_kind = "parallel"' in source


def test_simt_scope_defers_target_specific_physical_thread_limit():
    func = _simt_scope(2048)
    blocks = {
        block.name_hint: block
        for block in _collect_nodes(func.body, tvm.tirx.SBlock)
    }
    assert int(blocks["SimtVF"].annotations["tl.simt_threads"]) == 2048


@pytest.mark.parametrize("threads", [0, -1, 1 << 32])
def test_simt_scope_rejects_threads_outside_uint32(threads):
    with pytest.raises(ValueError):
        T.SimtVF(threads)


@pytest.mark.parametrize("threads", [True, 32.0, None, (32,), [32]])
def test_simt_scope_rejects_non_uint32_threads(threads):
    with pytest.raises(TypeError):
        T.SimtVF(threads)


@tilelang.testing.requires_package("mlir")
def test_simt_parallel_rejects_more_than_three_logical_axes():
    with pytest.raises(ValueError, match="one to three logical thread dimensions"):
        tilelang.lower(_simt_four_dimensional_logical_threads(), target="tile")


@tilelang.testing.requires_package("mlir")
def test_handwritten_scope_builder_matches_td_contract():
    from mlir import ir
    from mlir.dialects import arith

    from tilelangir import ScopeOp, ScopeYieldOp, configure_context

    with ir.Context() as context:
        configure_context(context)
        with ir.Location.unknown(context):
            scope = ScopeOp(mode="simt", threads=1024)
            body = ir.Block.create_at_start(scope.body)
            with ir.InsertionPoint(body):
                i32 = ir.IntegerType.get_signless(32)
                arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 0))
                ScopeYieldOp()

            assert scope.operation.verify()
            assert scope.threads == 1024
            assert str(scope.mode) == "#tilelang.scope_mode<simt>"
            assert "tilelang.scope_yield" in str(scope)


@tilelang.testing.requires_package("mlir")
def test_handwritten_scope_builder_rejects_mode_attribute_mismatches():
    from mlir import ir

    from tilelangir import ScopeOp, configure_context

    with ir.Context() as context:
        configure_context(context)
        with ir.Location.unknown(context):
            invalid_attributes = (
                {"mode": "simt"},
                {"mode": "simd", "threads": 32},
                {"mode": "default", "threads": 32},
                {"mode": "simt", "threads": 0},
                {"mode": "invalid"},
            )
            for kwargs in invalid_attributes:
                with pytest.raises(ValueError):
                    ScopeOp(**kwargs)


if __name__ == "__main__":
    tilelang.testing.main()
