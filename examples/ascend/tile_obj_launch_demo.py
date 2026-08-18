"""Stage-1 tile .o launch demo.

Workflow
--------
1. Compile the TileLang kernel to MLIR::

       python tile_obj_launch_demo.py --emit-mlir

   This prints the MLIR for ``vadd`` (the tile pipeline also prints the
   lowered TIR script).

2. Hand-compile that MLIR/CCE source into an object file (e.g.
   ``vadd_kernel.o``) with your toolchain.  The kernel symbol must be
   ``main`` (the PrimFunc name / ``global_symbol``) and the argument list
   must be (A, B, C) — three global-memory pointers — in that order.

3. Launch it::

       python tile_obj_launch_demo.py vadd_kernel.o

Requires a torch_npu environment and a tilelang build with src/tile enabled.
"""

import sys

import torch

import tilelang
import tilelang.language as T


def vadd(M=1024, N=256, block_M=32):
    num_blocks = M // block_M
    dtype = "float32"

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

            VL = 64

            with T.SimdVF():
                for r in range(0, block_M):
                    for i in range(0, N // VL):
                        a_frag = T.alloc_frag((VL,), dtype)
                        b_frag = T.alloc_frag((VL,), dtype)
                        c_frag = T.alloc_frag((VL,), dtype)

                        T.copy(a_shared[r, i * VL : (i + 1) * VL], a_frag)
                        T.copy(b_shared[r, i * VL : (i + 1) * VL], b_frag)

                        T.vadd(a_frag, b_frag, c_frag)

                        T.copy(c_frag, c_shared[r, i * VL : (i + 1) * VL])

            T.copy(c_shared, C[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def emit_mlir():
    program = vadd()
    artifact = tilelang.lower(program, target="tile")
    # The tile pipeline prints the lowered TIR script; kernel_source holds
    # the TileLangIR MLIR text that the hand-compilation consumes.
    print(artifact.kernel_source)


def launch(obj_path):
    from tilelang.opentile import compile_tile_obj

    M, N = 1024, 256
    program = vadd(M=M, N=N)
    # 1) lower 一次：拿到 MLIR（手工编译 .o 的依据）和可复用的 artifact。
    artifact = tilelang.lower(program, target="tile")
    # 2) 用同一个 artifact 绑定手工编译的 .o —— 不再重复 lower。
    kernel = compile_tile_obj(artifact, obj_path)

    a = torch.randn(M, N, device="npu", dtype=torch.float32)
    b = torch.randn(M, N, device="npu", dtype=torch.float32)
    c = torch.empty(M, N, device="npu", dtype=torch.float32)
    kernel(a, b, c)
    torch.npu.synchronize()

    ref = a + b
    torch.testing.assert_close(c, ref, rtol=1e-5, atol=1e-5)
    print("vadd tile .o launch: PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--emit-mlir":
        emit_mlir()
    elif len(sys.argv) > 1:
        launch(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(1)
