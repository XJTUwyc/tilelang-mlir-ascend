"""TileLangIR ``tilelang.vexpdif`` codegen tests.

Verifies that ``T.vexpdif`` in the DSL lowers to a ``linalg.generic`` structured
op whose body computes ``exp(src0 - src1)`` using ``arith.subf`` and
``math.exp``.  These are pure IR codegen tests: they call
``tilelang.lower(..., target="tile")`` and assert on the generated MLIR source
string.  No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vexpdif_kernel(
    VL: int = 64,
    src_dtype: str = "float32",
    dst_dtype: str = "float32",
):
    """Build a minimal kernel that applies ``dst = exp(src0 - src1)``.

    Two input fragments are copied from global tensors, the elementwise
    exponential-difference is written to a third fragment, and the result is
    copied back out.  Only the ``T.vexpdif`` call is exercised by the
    assertions; the surrounding ``T.copy`` ops provide a realistic context.
    """

    @T.prim_func
    def main(
        A: T.Tensor((1, VL), src_dtype),
        B: T.Tensor((1, VL), src_dtype),
        C: T.Tensor((1, VL), dst_dtype),
    ):
        with T.Kernel(1):
            a_shared = T.alloc_shared((1, VL), src_dtype)
            b_shared = T.alloc_shared((1, VL), src_dtype)
            c_shared = T.alloc_shared((1, VL), dst_dtype)
            T.copy(A[0:1, 0:VL], a_shared)
            T.copy(B[0:1, 0:VL], b_shared)

            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), src_dtype)
                b_frag = T.alloc_fragment((VL,), src_dtype)
                c_frag = T.alloc_fragment((VL,), dst_dtype)
                T.copy(a_shared[0, 0:VL], a_frag)
                T.copy(b_shared[0, 0:VL], b_frag)
                T.vexpdif(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0, 0:VL])

            T.copy(c_shared, C[0:1, 0:VL])

    return main


def _lower_to_tile_source(kernel) -> str:
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    artifact = tilelang.lower(kernel, target="tile")
    return str(artifact.kernel_source)


def _extract_vexpdif_block(source: str) -> str:
    """Return the ``linalg.generic`` body emitted for ``T.vexpdif``.

    ``vexpdif`` is the only op in the kernel that lowers to ``linalg.generic``
    with two inputs and one output, so the first ``linalg.generic`` match is
    unambiguously the one we want.
    """
    match = re.search(
        r"linalg\.generic\s*\{[^}]*\}\s*ins\([^)]*\)\s*outs\([^)]*\)\s*\{[^}]*\}",
        source, re.DOTALL,
    )
    assert match is not None, (
        "Expected a `linalg.generic` op in the generated MLIR, but none was found.\n"
        f"Source:\n{source}"
    )
    return match.group(0)


def _assert_body_op(block: str, op_name: str) -> None:
    """Assert that ``op_name`` (e.g. ``arith.subf``) appears in the generic body."""
    assert op_name in block, (
        f"Expected `{op_name}` in the vexpdif body, but it is missing.\n"
        f"Block:\n{block}"
    )


def _assert_indexing_map_count(block: str, count: int) -> None:
    """Assert that the generic op declares exactly ``count`` affine maps."""
    # The indexing_maps attribute prints as `{indexing_maps = [#map, #map, ...]}`.
    match = re.search(r"indexing_maps\s*=\s*\[([^\]]*)\]", block)
    assert match is not None, f"Could not find indexing_maps in block:\n{block}"
    map_count = match.group(1).count("#map")
    assert map_count == count, (
        f"Expected {count} indexing maps (src0, src1, dst), got {map_count}.\n"
        f"Block:\n{block}"
    )


def _assert_parallel_iterator(block: str) -> None:
    """Assert that the generic op uses only parallel iterators."""
    # iterator_types prints as `iterator_types = ["parallel", "parallel", ...]`.
    match = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', block)
    assert match is not None, f"Could not find iterator_types in block:\n{block}"
    types = match.group(1)
    assert "reduction" not in types, (
        f"Expected only parallel iterators, but found reduction.\nBlock:\n{block}"
    )


def test_vexpdif_emits_linalg_generic():
    """``T.vexpdif`` must lower to a ``linalg.generic`` structured op."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    assert "linalg.generic" in block


def test_vexpdif_body_uses_arith_subf():
    """The generic body must compute the difference with ``arith.subf``."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "arith.subf")


def test_vexpdif_body_uses_math_exp():
    """The generic body must apply ``math.exp`` to the difference."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "math.exp")


def test_vexpdif_body_uses_linalg_yield():
    """The generic body must terminate with ``linalg.yield``."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "linalg.yield")


def test_vexpdif_has_three_indexing_maps():
    """Two inputs (src0, src1) plus one output => three affine maps."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    _assert_indexing_map_count(block, 3)


def test_vexpdif_uses_parallel_iterators():
    """vexpdif is elementwise => all iterators must be parallel."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    _assert_parallel_iterator(block)


def test_vexpdif_subf_before_exp():
    """``math.exp`` must take the ``arith.subf`` result as its operand."""
    source = _lower_to_tile_source(_vexpdif_kernel())
    block = _extract_vexpdif_block(source)
    # MLIR prints `%result = arith.subf %lhs, %rhs : f32`, so the result is the
    # SSA value before `=`; math.exp then consumes that same value.
    subf_match = re.search(r"(%\S+)\s*=\s*arith\.subf", block)
    assert subf_match is not None, f"arith.subf not found in block:\n{block}"
    subf_result = subf_match.group(1)
    exp_match = re.search(r"math\.exp\s+(\S+)", block)
    assert exp_match is not None, f"math.exp not found in block:\n{block}"
    exp_operand = exp_match.group(1)
    assert exp_operand == subf_result, (
        f"math.exp operand ({exp_operand}) should be the arith.subf result "
        f"({subf_result}).\nBlock:\n{block}"
    )


def test_vexpdif_float16():
    """vexpdif on float16 fragments must lower correctly."""
    source = _lower_to_tile_source(
        _vexpdif_kernel(src_dtype="float16", dst_dtype="float16")
    )
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "arith.subf")
    _assert_body_op(block, "math.exp")
    assert "f16" in block, f"Expected f16 element type in block:\n{block}"


def test_vexpdif_mixed_element_types_raises():
    """Mixed src/dst element types must be rejected by MLIR verification.

    vexpdif computes exp(src0 - src1) on a single element type; we do not
    pre-check this (mirroring _emit_vadd's reliance on linalg.add), so the
    error surfaces from MLIR rather than from our own guard.
    """
    import pytest

    with pytest.raises(Exception):
        _lower_to_tile_source(
            _vexpdif_kernel(src_dtype="float32", dst_dtype="float16")
        )


def test_vexpdif_integer_operands_raises():
    """Integer operands must be rejected: arith.subf and math.exp are float-only.

    As with the mixed-type case, this is caught by MLIR verification rather
    than an explicit guard in _emit_vexpdif.
    """
    import pytest

    with pytest.raises(Exception):
        _lower_to_tile_source(
            _vexpdif_kernel(src_dtype="int32", dst_dtype="int32")
        )


def _vexpdif_nd_kernel(shape, src_dtype="float32", dst_dtype="float32"):
    """Build a kernel that applies vexpdif on N-dimensional fragments.

    Unlike ``_vexpdif_kernel`` (which is 1-D and wraps the op in ``T.copy``
    traffic for realism), this builder takes an arbitrary ``shape`` tuple and
    operates directly on fragments, so we can probe higher-rank lowering and
    shape/rank mismatch errors concisely.
    """

    @T.prim_func
    def main(
        A: T.Tensor(shape, src_dtype),
        B: T.Tensor(shape, src_dtype),
        C: T.Tensor(shape, dst_dtype),
    ):
        with T.Kernel(1):
            with T.SimdVF():
                af = T.alloc_fragment(shape, src_dtype)
                bf = T.alloc_fragment(shape, src_dtype)
                cf = T.alloc_fragment(shape, dst_dtype)
                T.vexpdif(af, bf, cf)

    return main


def test_vexpdif_bfloat16():
    """bf16 operands must lower like other float types."""
    source = _lower_to_tile_source(
        _vexpdif_nd_kernel((64,), src_dtype="bfloat16", dst_dtype="bfloat16")
    )
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "arith.subf")
    _assert_body_op(block, "math.exp")
    assert "bf16" in block, f"Expected bf16 element type in block:\n{block}"


def test_vexpdif_3d_operands():
    """3-D operands must produce three parallel iterators (rank-3 elementwise)."""
    source = _lower_to_tile_source(
        _vexpdif_nd_kernel((2, 4, 8), src_dtype="float32", dst_dtype="float32")
    )
    block = _extract_vexpdif_block(source)
    _assert_body_op(block, "arith.subf")
    _assert_body_op(block, "math.exp")
    # Three dims => three parallel iterators.
    match = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', block)
    assert match is not None, f"Could not find iterator_types in block:\n{block}"
    assert match.group(1).count("parallel") == 3, (
        f"Expected 3 parallel iterators for 3-D operands, got: {match.group(1)}"
    )


def test_vexpdif_rank_mismatch_raises():
    """Operands of different ranks must be rejected by MLIR verification."""
    import pytest

    @T.prim_func
    def main(
        A: T.Tensor((4, 16), "float32"),
        B: T.Tensor((64,), "float32"),
        C: T.Tensor((4, 16), "float32"),
    ):
        with T.Kernel(1):
            with T.SimdVF():
                af = T.alloc_fragment((4, 16), "float32")
                bf = T.alloc_fragment((64,), "float32")
                cf = T.alloc_fragment((4, 16), "float32")
                T.vexpdif(af, bf, cf)

    with pytest.raises(Exception):
        _lower_to_tile_source(main)


def test_vexpdif_shape_mismatch_raises():
    """Same rank but mismatched shapes must be rejected.

    We do not check shapes ourselves (linalg.generic does), so the error type
    is MLIR's ``MLIRError`` rather than our ``ValueError``; the test only
    asserts that lowering fails rather than emitting invalid IR.
    """
    import pytest

    @T.prim_func
    def main(
        A: T.Tensor((4, 16), "float32"),
        B: T.Tensor((8, 32), "float32"),
        C: T.Tensor((4, 16), "float32"),
    ):
        with T.Kernel(1):
            with T.SimdVF():
                af = T.alloc_fragment((4, 16), "float32")
                bf = T.alloc_fragment((8, 32), "float32")
                cf = T.alloc_fragment((4, 16), "float32")
                T.vexpdif(af, bf, cf)

    with pytest.raises(Exception):
        _lower_to_tile_source(main)


if __name__ == "__main__":
    tilelang.testing.main()
