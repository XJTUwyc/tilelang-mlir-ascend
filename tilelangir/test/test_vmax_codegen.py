"""TileLangIR ``tl.tileop.vmax`` codegen tests.

Verifies that ``T.vmax`` in the DSL lowers to a ``tilelang.vmax`` custom
dialect op whose ``dtype`` attribute preserves the frontend TIRX dtype name
(float32 / uint32 / int32), which carries the signedness the signless memref
element type drops.  These are pure IR codegen tests: they call
``tilelang.lower(..., target="tile")`` and assert on the generated MLIR source
string.  No Ascend hardware required.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _vmax_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
    dtype: str = "float32",
):
    """Build a minimal vmax kernel ``C = elementwise_max(A, B)``."""
    num_blocks = M // block_M
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


def _extract_vmax_op(source: str) -> str:
    """Return the ``tilelang.vmax`` operation emitted for ``T.vmax``."""
    match = re.search(r'"tilelang\.vmax"\([^)]*\)[^{]*\{[^}]*\}[^:]*:[^)]*\)',
                      source)
    assert match is not None, (
        "Expected a ``tilelang.vmax`` op in the generated MLIR, but none was "
        f"found.\nSource:\n{source}"
    )
    return match.group(0)


def test_vmax_basic_lowers_to_tilelang_vmax():
    """C = max(A, B) must emit at least one ``tilelang.vmax``."""
    source = _lower_to_tile_source(_vmax_kernel())
    assert "tilelang.vmax" in source


def test_vmax_operand_has_fragment_address_space():
    """The vmax op's memref operands should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vmax_kernel())
    op_text = _extract_vmax_op(source)
    assert "memref<64xf32, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )


def test_vmax_float_dtype_attr():
    """float32 vmax must carry ``dtype = "float32"``."""
    source = _lower_to_tile_source(_vmax_kernel())
    op_text = _extract_vmax_op(source)
    assert 'dtype = "float32"' in op_text, (
        f"Expected dtype = \"float32\".\nOp text:\n{op_text}"
    )


def test_vmax_unsigned_dtype_attr():
    """uint32 vmax must carry ``dtype = "uint32"`` (unsigned signedness)."""
    source = _lower_to_tile_source(_vmax_kernel(dtype="uint32"))
    op_text = _extract_vmax_op(source)
    assert 'dtype = "uint32"' in op_text, (
        f"Expected dtype = \"uint32\".\nOp text:\n{op_text}"
    )


def test_vmax_signed_dtype_attr():
    """int32 vmax must carry ``dtype = "int32"`` (signed signedness)."""
    source = _lower_to_tile_source(_vmax_kernel(dtype="int32"))
    op_text = _extract_vmax_op(source)
    assert 'dtype = "int32"' in op_text, (
        f"Expected dtype = \"int32\".\nOp text:\n{op_text}"
    )


def test_vmax_no_linalg_generic():
    """vmax must no longer lower to linalg.generic."""
    source = _lower_to_tile_source(_vmax_kernel())
    assert "linalg.generic" not in source


if __name__ == "__main__":
    tilelang.testing.main()
