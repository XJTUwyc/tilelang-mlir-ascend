"""TileLangIR ``tl.tileop.vmuls`` codegen tests.
 
Verifies that ``T.vmuls`` in the DSL lowers to a ``linalg.generic`` op with
the scalar captured as an SSA value from the parent scope (no extra memref
alloc for the scalar). These are pure IR codegen tests: they call
``tilelang.lower(..., target="tile")`` and assert on the generated MLIR source
string. No Ascend hardware required.
"""
 
import re
 
import tilelang
import tilelang.language as T
import tilelang.testing
 
 
def _vmuls_kernel(
    M: int = 1024,
    N: int = 256,
    block_M: int = 32,
    scalar_val: float = 2.0,
):
    """Build a minimal vmuls kernel ``C = A * scalar`` (one block per row-tile).
 
    Only the ``T.vmuls`` call is exercised; the surrounding ``T.copy`` ops are
    intentionally left in place so the TIRX IR shape matches a realistic kernel.
    """
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
 
                        T.vmuls(a_frag, scalar_val, c_frag)
 
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
 
 
def test_vmuls_basic_lowers_to_linalg_generic():
    """C = A * scalar must emit at least one ``linalg.generic``."""
    source = _lower_to_tile_source(_vmuls_kernel())
    _assert_linalg_generic(source)
 
 
def test_vmuls_operand_has_fragment_address_space():
    """Both ``ins`` and ``outs`` memrefs should live in address space 2 (fragment)."""
    source = _lower_to_tile_source(_vmuls_kernel())
    match = _assert_linalg_generic(source)
    op_text = match.group(0)
    # Fragment buffers have address space 2 in the MLIR output.
    assert "memref<64xf32, strided<[1]>, 2>" in op_text, (
        f"Expected fragment memref with address space 2, got:\n{op_text}"
    )
 
 
def test_vmuls_scalar_constant_is_present():
    """The scalar constant (e.g. 2.0) must appear as an ``arith.constant``."""
    source = _lower_to_tile_source(_vmuls_kernel(scalar_val=2.0))
    assert "arith.constant 2.000000e+00 : f32" in source, (
        "Expected scalar constant 2.0 in the generated MLIR.\n"
        f"Source:\n{source}"
    )
 
 
def test_vmuls_body_has_arith_mulf():
    """The ``linalg.generic`` body must contain ``arith.mulf``."""
    source = _lower_to_tile_source(_vmuls_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    assert "arith.mulf" in body, (
        "Expected ``arith.mulf`` in the linalg.generic body.\n"
        f"Body:\n{body}"
    )
 
 
def test_vmuls_body_yields_mulf_result():
    """The ``linalg.generic`` body must contain ``linalg.yield`` with the mulf result."""
    source = _lower_to_tile_source(_vmuls_kernel())
    match = _assert_linalg_generic(source)
    body = match.group(0)
    # The yield should follow the mulf within the body.
    yield_match = re.search(r"linalg\.yield\s+%[^:\s]+", body)
    assert yield_match is not None, (
        "Expected ``linalg.yield`` with the mulf result in the generic body.\n"
        f"Body:\n{body}"
    )
 
 
def test_vmuls_indexing_maps_are_identity():
    """The ``indexing_maps`` should be ``#map = affine_map<(d0) -> (d0)>``."""
    source = _lower_to_tile_source(_vmuls_kernel())
    # The identity affine map for 1D: (d0) -> (d0)
    assert "affine_map<(d0) -> (d0)>" in source, (
        "Expected identity affine map in the generated MLIR.\n"
        f"Source:\n{source}"
    )
 
 
def test_vmuls_iterator_types_are_parallel():
    """The ``iterator_types`` should be ``[\"parallel\"]``."""
    source = _lower_to_tile_source(_vmuls_kernel())
    assert 'iterator_types = ["parallel"]' in source, (
        "Expected iterator_types = [\"parallel\"] in linalg.generic.\n"
        f"Source:\n{source}"
    )
 
 
def test_vmuls_different_scalar_values():
    """Different scalar values must produce different constants in the MLIR."""
    source_2 = _lower_to_tile_source(_vmuls_kernel(scalar_val=2.0))
    assert "arith.constant 2.000000e+00 : f32" in source_2
 
    source_3 = _lower_to_tile_source(_vmuls_kernel(scalar_val=3.0))
    assert "arith.constant 3.000000e+00 : f32" in source_3
 
 
if __name__ == "__main__":
    tilelang.testing.main()