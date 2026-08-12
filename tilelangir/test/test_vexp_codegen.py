"""TileLangIR ``tl.tileop.vexp`` codegen tests.

Verifies that ``T.vexp`` in the DSL lowers to a ``linalg.generic`` op with
``arith.subf`` and ``math.exp`` in the body. These are pure IR codegen tests:
they call ``tilelang.lower(..., target="tile")`` and assert on the generated
MLIR source string. No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vexp_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
):
    """Build a minimal vexp kernel ``C = exp(A - B)`` (one block per row-tile)."""
    num_blocks = M // block_M
    dtype = "float32"
    VL = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            b_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M, N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)
            T.copy(B[bx * block_M : (bx + 1) * block_M, 0:N], b_shared)

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        b_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)
                        T.copy(b_shared[r, i * VL : (i + 1) * VL], b_frag)

                        T.vexp(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


# Regex that matches a ``linalg.generic`` op and captures its full content.
_LINALG_GENERIC_RE = re.compile(
    r"linalg\.generic\s*\{[^}]*\}\s*ins\([^)]*\)\s*outs\([^)]*\)\s*\{[^}]*\}",
    re.MULTILINE | re.DOTALL,
)


def _assert_linalg_generic(source: str) -> re.Match:
    """Assert that the source contains at least one ``linalg.generic`` op."""
    match = _LINALG_GENERIC_RE.search(source)
    assert match is not None, (
        "Expected a ``linalg.generic`` op in the generated MLIR, but none was found.\n"
        f"Source:\n{source}"
    )
    return match


def test_vexp_basic_lowers_to_linalg_generic():
    """C = exp(A - B) must emit at least one ``linalg.generic``."""
    source = _lower_to_tile_source(_vexp_kernel())
    _assert_linalg_generic(source)


def test_vexp_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vexp_kernel())
    match = _assert_linalg_generic(source)
    op_text = match.group(0)
    assert "memref<64xf32, strided<[1]>, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )


def test_vexp_body_has_arith_subf():
    """The ``linalg.generic`` body must contain ``arith.subf``."""
    source = _lower_to_tile_source(_vexp_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "arith.subf" in body, (
        "Expected ``arith.subf`` in the linalg.generic body.\n"
        f"Body:\n{body}"
    )


def test_vexp_body_has_math_exp():
    """The ``linalg.generic`` body must contain ``math.exp``."""
    source = _lower_to_tile_source(_vexp_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "math.exp" in body, (
        "Expected ``math.exp`` in the linalg.generic body.\n"
        f"Body:\n{body}"
    )


def test_vexp_body_yields_exp_result():
    """The ``linalg.generic`` body must contain ``linalg.yield`` with the exp result."""
    source = _lower_to_tile_source(_vexp_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    yield_match = re.search(r"linalg\.yield\s+%[^:\s]+", body)
    assert yield_match is not None, (
        "Expected ``linalg.yield`` with the exp result in the generic body.\n"
        f"Body:\n{body}"
    )


def test_vexp_indexing_maps_are_identity():
    """The ``indexing_maps`` should be ``#map = affine_map<(d0) -> (d0)>``."""
    source = _lower_to_tile_source(_vexp_kernel())
    assert "affine_map<(d0) -> (d0)>" in source, (
        "Expected identity affine map in the generated MLIR.\n"
        f"Source:\n{source}"
    )


def test_vexp_iterator_types_are_parallel():
    """The ``iterator_types`` should be ``[\"parallel\"]``."""
    source = _lower_to_tile_source(_vexp_kernel())
    assert 'iterator_types = ["parallel"]' in source, (
        "Expected iterator_types = [\"parallel\"] in linalg.generic.\n"
        f"Source:\n{source}"
    )


if __name__ == "__main__":
    tilelang.testing.main()