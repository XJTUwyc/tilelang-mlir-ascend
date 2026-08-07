"""TileLangIR ``tilelang.gemm`` codegen tests.

Verifies that ``T.gemm`` in the DSL lowers to a ``tilelang.gemm`` MLIR op with
the expected operands and UnitAttrs (transpose_a / transpose_b / clear_accum).
These are pure IR codegen tests: they call ``tilelang.lower(..., target="tile")``
and assert on the generated MLIR source string. No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _gemm_kernel(
    M: int = 128,
    N: int = 128,
    K: int = 128,
    block_M: int = 32,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
):
    """Build a minimal GEMM kernel ``C = A @ B`` (one block per row-tile of C).

    Only the ``T.gemm`` call is exercised; the surrounding ``T.copy`` ops are
    not yet translated by the TileLangIR translator and are intentionally left
    in place so the TIRX IR shape matches a realistic kernel.

    When ``transpose_A``/``transpose_B`` is set, the shared-memory tile shape
    must match the transposed operand: e.g. ``transpose_A`` swaps the two dims
    of ``A_shared`` so that ``A_shared.shape[-1] == block_M`` (== C's M dim),
    as required by ``T.gemm``'s shape check in
    [tilelang/language/gemm_op.py](../../tilelang/language/gemm_op.py).
    """
    dtype = "float16"
    accum_dtype = "float32"
    num_blocks = M // block_M
    a_tile_shape = (K, block_M) if transpose_A else (block_M, K)
    b_tile_shape = (N, K) if transpose_B else (K, N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            A_shared = T.alloc_shared(a_tile_shape, dtype)
            B_shared = T.alloc_shared(b_tile_shape, dtype)
            C_local = T.alloc_fragment((block_M, N), accum_dtype)

            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:K], A_shared)
            T.copy(B[0:K, 0:N], B_shared)

            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
                clear_accum=clear_accum,
            )

            T.copy(C_local, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    artifact = tilelang.lower(kernel, target="tile")
    return artifact.kernel_source


# Regex that matches a ``tilelang.gemm`` op and captures its operand type list.
# MLIR prints custom ops as `"op"(...) {attr_dict} : (types) -> ()`; the
# attribute dict is optional, hence `\s*(?:\{[^}]*\})?` between operands and
# the type signature.
_GEMM_OP_RE = re.compile(
    r'"tilelang\.gemm"\([^)]*\)\s*(?:\{[^}]*\})?\s*:\s*\(([^)]*)\)\s*->\s*\(\)',
    re.MULTILINE,
)


def _assert_gemm_op(source: str) -> re.Match:
    match = _GEMM_OP_RE.search(source)
    assert match is not None, (
        "Expected a `tilelang.gemm` op in the generated MLIR, but none was found.\n"
        f"Source:\n{source}"
    )
    return match


def _assert_attr(source: str, attr: str, present: bool) -> None:
    """Assert whether a given UnitAttr name appears on the gemm op.

    MLIR prints UnitAttrs as a bare key inside the op's attribute dictionary,
    e.g. ``{transpose_a}``. The attribute names are unique enough that a simple
    substring check on the whole source is safe.
    """
    # Match the full gemm op line including its optional attr dict.
    full_match = re.search(
        r'"tilelang\.gemm"\([^)]*\)\s*(?:\{[^}]*\})?', source
    )
    assert full_match is not None, "No tilelang.gemm op found when checking attrs"
    op_line = full_match.group(0)
    if present:
        assert attr in op_line, (
            f"Expected attribute {attr!r} in gemm op, but it is missing.\nOp: {op_line}"
        )
    else:
        assert attr not in op_line, (
            f"Did not expect attribute {attr!r} in gemm op, but it is present.\nOp: {op_line}"
        )


def test_gemm_basic_codegens_tilelang_gemm_op():
    """C = A @ B with default flags must emit exactly one ``tilelang.gemm``."""
    source = _lower_to_tile_source(_gemm_kernel())
    _assert_gemm_op(source)


def test_gemm_operands_have_expected_address_spaces():
    """A/B should live in address space 1 (shared), C in space 2 (fragment)."""
    source = _lower_to_tile_source(_gemm_kernel())
    match = _assert_gemm_op(source)
    operand_types = match.group(1)
    # A and B are shared buffers -> address space 1
    assert "memref<32x128xf16, strided<[128, 1]>, 1>" in operand_types, (
        f"Expected A operand in address space 1 (shared), got: {operand_types}"
    )
    # C is a fragment buffer -> address space 2
    assert "memref<32x128xf32, strided<[128, 1]>, 2>" in operand_types, (
        f"Expected C operand in address space 2 (fragment), got: {operand_types}"
    )


def test_gemm_default_has_no_unit_attrs():
    """With transpose/clear_accum all False, no UnitAttr should be emitted."""
    source = _lower_to_tile_source(_gemm_kernel())
    _assert_attr(source, "transpose_a", present=False)
    _assert_attr(source, "transpose_b", present=False)
    _assert_attr(source, "clear_accum", present=False)


def test_gemm_clear_accum_emits_unit_attr():
    """``clear_accum=True`` must emit the ``clear_accum`` UnitAttr."""
    source = _lower_to_tile_source(_gemm_kernel(clear_accum=True))
    _assert_gemm_op(source)
    _assert_attr(source, "clear_accum", present=True)


def test_gemm_transpose_a_emits_unit_attr():
    """``transpose_A=True`` must emit the ``transpose_a`` UnitAttr."""
    source = _lower_to_tile_source(_gemm_kernel(transpose_A=True))
    _assert_gemm_op(source)
    _assert_attr(source, "transpose_a", present=True)
    # Other flags still off.
    _assert_attr(source, "transpose_b", present=False)
    _assert_attr(source, "clear_accum", present=False)


def test_gemm_transpose_b_emits_unit_attr():
    """``transpose_B=True`` must emit the ``transpose_b`` UnitAttr."""
    source = _lower_to_tile_source(_gemm_kernel(transpose_B=True))
    _assert_gemm_op(source)
    _assert_attr(source, "transpose_b", present=True)
    _assert_attr(source, "transpose_a", present=False)
    _assert_attr(source, "clear_accum", present=False)


if __name__ == "__main__":
    tilelang.testing.main()
