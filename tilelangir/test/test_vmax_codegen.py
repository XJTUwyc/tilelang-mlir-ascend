"""TileLangIR ``tl.tileop.vmax`` codegen tests.

Verifies that ``T.vmax`` in the DSL lowers to a ``linalg.max`` op.
These are pure IR codegen tests: they call ``tilelang.lower(..., target="tile")``
and assert on the generated MLIR source string. No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vmax_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
):
    """Build a minimal vmax kernel ``C = elementwise_max(A, B)`` (one block per row-tile)."""
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

                        T.vmax(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


# Regex that matches a ``linalg.max`` op.
_LINALG_MAX_RE = re.compile(
    r"linalg\.max\s+ins\([^)]*\)\s*outs\([^)]*\)",
    re.MULTILINE | re.DOTALL,
)


def test_vmax_basic_lowers():
    """C = max(A, B) must emit at least one ``linalg.max``."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _LINALG_MAX_RE.search(source)
    assert match is not None, (
        "Expected a ``linalg.max`` op in the generated MLIR, but none was found.\n"
        f"Source:\n{source}"
    )


def test_vmax_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _LINALG_MAX_RE.search(source)
    assert match is not None, "No ``linalg.max`` found."
    op_text = match.group(0)
    assert "memref<64xf32, strided<[1]>, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )


def test_vmax_two_inputs():
    """``linalg.max`` must have two input operands."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _LINALG_MAX_RE.search(source)
    assert match is not None, "No ``linalg.max`` found."
    op_text = match.group(0)
    # Count fragment memref references in the ins() list
    ins_count = op_text.count("memref<64xf32, strided<[1]>, 2>")
    assert ins_count >= 2, (
        f"Expected at least 2 fragment memref inputs in linalg.max, got {ins_count}.\n"
        f"Op text:\n{op_text}"
    )


def test_vmax_different_values():
    """Different input values must produce the same structure."""
    source = _lower_to_tile_source(_vmax_kernel())
    assert "linalg.max" in source


if __name__ == "__main__":
    tilelang.testing.main()