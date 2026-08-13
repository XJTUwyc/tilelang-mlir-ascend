import tilelang
import tilelang.language as T

def vmul(M=1024, N=256, block_M=32):
    num_blocks = M // block_M
    dtype="float32"

    @T.prim_func
    def main(
        A: T.Buffer((M, N), dtype),
        B: T.Buffer((M, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            b_shared = T.alloc_shared((block_M, N), dtype)
            c_shared = T.alloc_shared((block_M, N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)
            T.copy(B[bx * block_M : (bx + 1) * block_M, 0:N], b_shared)
            
            VL=64 

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        b_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)
                        T.copy(b_shared[r, i * VL : (i + 1) * VL], b_frag)

                        T.vmul(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main

if __name__ == "__main__":
    program = vmul()
    result = tilelang.lower(program, target="tile")
    mlir_source = result.kernel_source
    print(mlir_source)
    # Check that the lowered MLIR contains the expected operations
    assert mlir_source is not None and len(mlir_source) > 0, "Lowered MLIR output is empty"
    assert "func.func" in mlir_source, "Missing func.func in lowered output"
    assert "tilelang.scope" in mlir_source, "Missing tilelang.scope in lowered output"
    assert "linalg.mul" in mlir_source, "Missing linalg.mul in lowered output"
    print("All checks passed!")
