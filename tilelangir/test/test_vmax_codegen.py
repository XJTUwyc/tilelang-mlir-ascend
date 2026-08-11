"""TileLangIR ``tl.tileop.vmax`` codegen tests.

Verifies that ``T.vmax`` in the DSL lowers to a ``linalg.generic`` op whose
body uses the dtype-appropriate ``arith`` max op:

- float   -> ``arith.maxf``
- unsigned integer -> ``arith.maxui``
- signed integer   -> ``arith.maxsi``

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
    dtype: str = "float32",
):
    """Build a minimal vmax kernel ``C = elementwise_max(A, B)`` (one block per row-tile)."""
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


def test_vmax_basic_lowers_to_linalg_generic():
    """C = max(A, B) must emit at least one ``linalg.generic``."""
    source = _lower_to_tile_source(_vmax_kernel())
    _assert_linalg_generic(source)


def test_vmax_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _assert_linalg_generic(source)
    op_text = match.group(0)
    # Fragment buffers have address space 2 in the MLIR output.
    assert "memref<64xf32, strided<[1]>, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )


def test_vmax_two_inputs():
    """``linalg.generic`` must have two fragment memref inputs."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _assert_linalg_generic(source)
    op_text = match.group(0)
    # Count fragment memref references in the ins() list.
    ins_count = op_text.count("memref<64xf32, strided<[1]>, 2>")
    assert ins_count >= 2, (
        f"Expected at least 2 fragment memref inputs in linalg.generic, got {ins_count}.\n"
        f"Op text:\n{op_text}"
    )


def test_vmax_body_has_arith_maxf():
    """The ``linalg.generic`` body must contain ``arith.maxf`` for float dtype."""
    source = _lower_to_tile_source(_vmax_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "arith.maximumf" in body, (
        "Expected ``arith.maxf`` in the linalg.generic body.\n"
        f"Body:\n{body}"
    )


def test_vmax_unsigned_dtype_uses_maxui():
    """Unsigned integer vmax must emit ``arith.maxui`` in the body."""
    source = _lower_to_tile_source(_vmax_kernel(dtype="uint32"))
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "arith.maxui" in body, (
        "Expected ``arith.maxui`` for unsigned vmax.\n"
        f"Body:\n{body}"
    )
    assert "arith.maximumf" not in body, (
        "Unsigned vmax should not emit ``arith.maxf``.\n"
        f"Body:\n{body}"
    )


def test_vmax_signed_dtype_uses_maxsi():
    """Signed integer vmax must emit ``arith.maxsi`` in the body."""
    source = _lower_to_tile_source(_vmax_kernel(dtype="int32"))
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "arith.maxsi" in body, (
        "Expected ``arith.maxsi`` for signed vmax.\n"
        f"Body:\n{body}"
    )
    assert "arith.maximumf" not in body, (
        "Signed vmax should not emit ``arith.maxf``.\n"
        f"Body:\n{body}"
    )


def test_vmax_different_values():
    """Different input values must produce the same structure."""
    source = _lower_to_tile_source(_vmax_kernel())
    assert "linalg.generic" in source


if __name__ == "__main__":
    tilelang.testing.main()
