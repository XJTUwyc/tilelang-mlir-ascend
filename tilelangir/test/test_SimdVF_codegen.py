"""TileLangIR ``tilelang.scope`` codegen tests for ``T.SimdVF``."""

from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing


def _kernel_source(artifact) -> str:
    source = getattr(artifact, "kernel_source", None)
    if source:
        return str(source)
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is not None and hasattr(device_mod, "inspect_source"):
        inspected = device_mod.inspect_source()
        if inspected:
            return str(inspected)
    raise AssertionError("lower(target='tile') did not produce inspectable MLIR source")


def _static_simdvf_kernel(M=32, N=256, VL=64, dtype="float32"):
    """Shared->frag->shared pipeline wrapped in a single ``T.SimdVF()`` scope."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1):
            a_shared = T.alloc_shared((M, N), dtype)
            b_shared = T.alloc_shared((M, N), dtype)
            c_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], a_shared)
            T.copy(B[0:M, 0:N], b_shared)

            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), dtype)
                b_frag = T.alloc_fragment((VL,), dtype)
                c_frag = T.alloc_fragment((VL,), dtype)

                T.copy(a_shared[0, 0:VL], a_frag)
                T.copy(b_shared[0, 0:VL], b_frag)
                T.vadd(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0, 0:VL])

            T.copy(c_shared, C[0:M, 0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_simdvf_emits_tilelang_scope():
    artifact = tilelang.lower(_static_simdvf_kernel(), target="tile")
    mlir = _kernel_source(artifact)
    print(f"mlr:\n{mlir}")
    assert "tilelang.scope" in mlir
    assert "simd_attr" in mlir
    assert 'simd_attr = "simd"' in mlir
    assert "tilelang.copy" in mlir
    assert "linalg.add" in mlir
    # The SimdVF SBlock's reads are serialized onto the scope op. The writes
    # list is empty here, so no writes attribute must be emitted.
    assert 'reads = "a_shared[0:1, 0:1], b_shared[0:1, 0:1], c_shared[0:1, 0:1]"' in mlir
    assert "writes =" not in mlir


def _frag_only_simdvf_kernel(N=256, VL=64, dtype="float32"):
    """Kernel whose only fragment allocations live inside the SimdVF scope."""

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((N,), dtype)
            c_shared = T.alloc_shared((N,), dtype)
            T.copy(A[0:N], a_shared)
            with T.SimdVF():
                a_frag = T.alloc_fragment((VL,), dtype)
                b_frag = T.alloc_fragment((VL,), dtype)
                c_frag = T.alloc_fragment((VL,), dtype)

                T.copy(a_shared[0:VL], a_frag)
                T.copy(a_shared[VL : 2 * VL], b_frag)
                T.vadd(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0:VL])
            T.copy(c_shared, C[0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_simdvf_scope_wraps_fragment_allocs_and_body():
    artifact = tilelang.lower(_frag_only_simdvf_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.scope" in mlir
    # Fragment allocations (local.fragment => address-space 2) and the vadd
    # body are emitted inside the tilelang.scope region, i.e. after the scope
    # keyword. Outer shared allocs/copies may legitimately precede it.
    assert mlir.index("tilelang.scope") < mlir.index("memref.alloc() : memref<64xf32, 2>")
    assert mlir.index("tilelang.scope") < mlir.index("linalg.add")
    # a_frag, b_frag and c_frag must each produce a fragment allocation.
    assert mlir.count("memref.alloc() : memref<64xf32, 2>") == 3


def _loop_simdvf_kernel(M=32, N=256, VL=64, dtype="float32"):
    """SimdVF scope wrapping serial loops that copy shared rows to fragments."""

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((M, N), dtype)
            T.copy(A[0:M, 0:N], a_shared)
            with T.SimdVF():
                for r in T.serial(0, M):
                    for i in T.serial(0, N // VL):
                        frag = T.alloc_fragment((VL,), dtype)
                        T.copy(a_shared[r, i * VL : (i + 1) * VL], frag)
                        T.copy(frag, a_shared[r, i * VL : (i + 1) * VL])
            T.copy(a_shared, B[0:M, 0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_simdvf_nested_loops_emitted_inside_scope():
    artifact = tilelang.lower(_loop_simdvf_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.scope" in mlir
    assert "scf.for" in mlir
    # Loop bodies must live inside the tilelang.scope region.
    assert mlir.index("tilelang.scope") < mlir.index("scf.for")
    assert "tilelang.copy" in mlir


def _shared_const_simdvf_kernel(N=256, VL=64, dtype="float32"):
    """Share buffers and constants across the tilelang.scope region boundary.

    The a_shared buffer is allocated outside the scope, read inside it, and
    copied out afterwards. This exercises the value_map handling around the
    region: the enclosing block must not depend on values created inside it.
    """

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((N,), dtype)
            T.copy(A[0:N], a_shared)
            with T.SimdVF():
                frag = T.alloc_fragment((VL,), dtype)
                T.copy(a_shared[0:VL], frag)
            T.copy(a_shared, B[0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_simdvf_scope_boundary_reuse():
    artifact = tilelang.lower(_shared_const_simdvf_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.scope" in mlir
    # Both a copy before and after the scope are emitted.
    assert mlir.count("tilelang.copy") == 3


def _dynamic_simdvf_kernel(N=256, VL=64, dtype="float32"):
    """SimdVF scope whose loop bound is a dynamic scalar parameter.

    The auto-detected reads region then carries a symbolic extent; serializing
    it must not call ``int()`` on a non-constant expression.
    """

    @T.prim_func
    def main(n: T.int32, A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((N,), dtype)
            T.copy(A[0:N], a_shared)
            with T.SimdVF():
                frag = T.alloc_fragment((VL,), dtype)
                for i in T.serial(0, n):
                    T.copy(a_shared[i : i + VL], frag)
            T.copy(a_shared, B[0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_simdvf_dynamic_region_serialization():
    artifact = tilelang.lower(_dynamic_simdvf_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.scope" in mlir
    # Symbolic bounds are printed as expressions instead of crashing int().
    assert 'reads = "a_shared[0:0+n]"' in mlir


if __name__ == "__main__":
    tilelang.testing.main()
