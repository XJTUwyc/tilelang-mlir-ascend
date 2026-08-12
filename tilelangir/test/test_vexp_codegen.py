"""TileLangIR ``tl.tileop.vexp`` codegen tests.

Verifies that unary ``T.vexp`` lowers to the corresponding named
``linalg.exp`` operation. These are pure IR codegen tests and require no
Ascend hardware.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def _vexp_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
):
    """Build a minimal vexp kernel ``C = exp(A)`` (one block per row-tile)."""
    num_blocks = M // block_M
    dtype = "float32"
    VL = 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M, N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)
                        T.vexp(a_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


def test_vexp_basic_lowers_to_named_linalg_exp():
    """C = exp(A) must emit the corresponding named Linalg operation."""
    source = _lower_to_tile_source(_vexp_kernel())
    assert "linalg.exp" in source


def test_vexp_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vexp_kernel())
    assert "memref<64xf32, strided<[1]>, 2>" in source, (
        f"Expected fragment memref with address space 2, got:\n{source}"
    )


def test_vexp_does_not_emit_subtraction():
    """Unary vexp must not retain the old exponential-difference payload."""
    source = _lower_to_tile_source(_vexp_kernel())
    assert "arith.subf" not in source


if __name__ == "__main__":
    tilelang.testing.main()
