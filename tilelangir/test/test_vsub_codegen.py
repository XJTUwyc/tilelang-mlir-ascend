"""TileLangIR ``tl.tileop.vsub`` codegen tests.

Verifies that binary ``T.vsub`` lowers to the corresponding named
``linalg.sub`` operation, matching the ``T.vdiv`` lowering path. These are
pure IR codegen tests and require no Ascend hardware.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def _vsub_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
):
    """Build a minimal vsub kernel ``C = A - B`` (one block per row-tile)."""
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
                        T.vsub(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


def test_vsub_basic_lowers_to_named_linalg_sub():
    """C = A - B must emit the corresponding named Linalg operation."""
    source = _lower_to_tile_source(_vsub_kernel())
    assert "linalg.sub" in source


def test_vsub_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vsub_kernel())
    assert "memref<64xf32, 2>" in source, (
        f"Expected fragment memref with address space 2, got:\n{source}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
