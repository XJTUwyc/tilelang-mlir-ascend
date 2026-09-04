"""Vector broadcast kernel on Ascend NPU (Vector pipeline).

T.broadcast(A, C)

Broadcasts the (M,) column vector ``A`` to the (M, N) output ``C``, i.e.
``C[r, n] = A[r]`` for every column ``n``.  Inside the kernel this is done by
broadcasting an ``(block_M,)`` fragment to an ``(block_M, VL)`` fragment and
writing each VL-wide column chunk back to shared memory.

Uses SimdVF for the vector broadcast. No split_dim on T.copy and no T.For in
scope, matching the tilelangir test conventions.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def vbroadcast(M=1024, N=256, block_M=128, dtype="float32"):
    num_blocks = M // block_M
    VL = 64

    @T.prim_func
    def main(
        A: T.Tensor((M,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M,), dtype)
            c_shared = T.alloc_shared((block_M, N), dtype)

            T.copy(A[bx * block_M : (bx + 1) * block_M], a_shared)

            with T.SimdVF():
                a_frag = T.alloc_frag((block_M,), dtype)
                b_frag = T.alloc_frag((VL,), dtype)
                l_bc = T.alloc_frag((block_M, VL), dtype)

                T.copy(a_shared[0:block_M], a_frag)

                # l_bc[r, i] = a_frag[r] = A[bx * block_M + r]
                T.broadcast(a_frag, l_bc)
                T.broadcast(b_frag, l_bc)

                for i in range(0, N // VL):
                    T.copy(l_bc, c_shared[0:block_M, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


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


@tilelang.testing.requires_package("mlir")
def test_broadcast_emits_linalg_broadcast():
    artifact = tilelang.lower(vbroadcast(), target="tile")
    mlir = _kernel_source(artifact)

    assert "tilelang.scope" in mlir
    assert "linalg.broadcast" in mlir
    assert "dimensions = [1]" in mlir
    assert "dimensions = [0]" in mlir
    assert "tilelang.copy" in mlir
    # No tensor conversions should remain.
    assert "bufferization.to_tensor" not in mlir
    assert "tensor.generate" not in mlir
    assert "materialize_in_destination" not in mlir


if __name__ == "__main__": 
    tilelang.testing.main()
