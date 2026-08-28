"""Region squeeze, scf.for inner indices, and dynamic-size copy tests."""

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


def _inner_loop_copy_kernel(M=32, N=256, VL=64, dtype="float32"):
    """Shared→frag copy with loop-carried indices ``r`` / ``i`` (no bx)."""

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


def _dynamic_size_copy_kernel(N=32, dtype="float32"):
    """1-D copy whose region extent is the loop induction variable."""

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(1):
            a_shared = T.alloc_shared((N,), dtype)
            b_shared = T.alloc_shared((N,), dtype)
            T.copy(A[0:N], a_shared)
            for n in T.serial(1, N + 1):
                T.copy(a_shared[0:n], b_shared[0:n])
            T.copy(b_shared, B[0:N])

    return main


@tilelang.testing.requires_package("mlir")
def test_inner_loop_copy_emits_scf_for_and_squeezed_1d_view():
    artifact = tilelang.lower(_inner_loop_copy_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "scf.for" in mlir
    assert mlir.count("scf.for") >= 2
    assert "tilelang.scope" in mlir
    assert "mode = #tilelang.scope_mode<simd>" in mlir
    assert "tilelang.copy" in mlir

    # Golden-style squeeze: ``[1, VL]`` → ``memref<64xf32, ...>``, not ``1x64``.
    assert "memref<1x64xf32" not in mlir
    assert "memref<64xf32" in mlir
    assert "arith.muli" in mlir


@tilelang.testing.requires_package("mlir")
def test_dynamic_size_copy_emits_dynamic_memref_sizes():
    artifact = tilelang.lower(_dynamic_size_copy_kernel(), target="tile")
    mlir = _kernel_source(artifact)

    assert "scf.for" in mlir
    assert "tilelang.copy" in mlir
    # Dynamic extent becomes ``?`` in the result memref type.
    assert "memref<?xf32" in mlir
    assert "memref.reinterpret_cast" in mlir


if __name__ == "__main__":
    tilelang.testing.main()
