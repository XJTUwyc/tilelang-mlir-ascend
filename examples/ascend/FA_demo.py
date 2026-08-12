"""FlashAttention forward kernel on Ascend NPU using Mix kernel (Cube + Vector).

Online-softmax FlashAttention: O = softmax(Q @ K^T / sqrt(d)) @ V

Uses SimdVF for the online softmax (row-max, exp, sum, rescale).
"""

import math

import tilelang
import tilelang.language as T


def flash_attention(D=128, SEQ_LEN=4096):
    BR = 128
    BC = 128
    NUM_KV_BLOCKS = SEQ_LEN // BC
    dtype = "bfloat16"
    accum_dtype = "float32"

    scale = math.sqrt(1.0 / D) * 1.4426950408889634

    @T.prim_func
    def main(
        Q: T.Buffer((SEQ_LEN, D), dtype),
        K: T.Buffer((SEQ_LEN, D), dtype),
        V: T.Buffer((SEQ_LEN, D), dtype),
        O: T.Buffer((SEQ_LEN, D), accum_dtype),
    ):
        with T.Kernel(SEQ_LEN // BR) as bx:
            Q_shared = T.alloc_shared((BR, D), dtype)
            K_shared = T.alloc_shared((BC, D), dtype)
            V_shared = T.alloc_shared((D, BC), dtype)
            P_shared = T.alloc_shared((BR, BC), dtype)

            acc_s = T.alloc_fragment((BR, BC), accum_dtype)
            acc_o = T.alloc_fragment((BR, D), accum_dtype)
            
            S_ub = T.alloc_shared((BR // 2, BC), accum_dtype)
            P_ub = T.alloc_shared((BR // 2, BC), dtype)
            O_ub = T.alloc_shared((BR // 2, D), accum_dtype)
            O_tmp_ub = T.alloc_shared((BR // 2, D), accum_dtype)
            m_ub = T.alloc_shared((BR // 2,), accum_dtype)
            l_ub = T.alloc_shared((BR // 2,), accum_dtype)
            alpha_ub = T.alloc_shared((BR // 2,), accum_dtype)

            T.copy(Q[bx * BR : (bx + 1) * BR, 0:D], Q_shared)

            with T.SimdVF():
                T.fill(O_ub, 0.0)
                T.fill(l_ub, 0.0)
                T.fill(m_ub, -T.infinity(accum_dtype))

            for k in T.Pipelined(NUM_KV_BLOCKS, num_stages=2):
                T.copy(K[k * BC : (k + 1) * BC, 0:D], K_shared)
                T.copy(
                    V[k * BC : (k + 1) * BC, 0:D],
                    V_shared,
                    transpose=True,
                )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                T.copy(acc_s, S_ub, split_dim=-1)

                scale_ln2 = scale * 0.6931471805599453
                ROWS = BR // 2
                VL = 64

                with T.SimdVF():
                    S_frag_0 = T.alloc_frag((VL,), accum_dtype)
                    S_frag_1 = T.alloc_frag((VL,), accum_dtype)
                    P_frag_0 = T.alloc_frag((VL,), accum_dtype)
                    P_frag_1 = T.alloc_frag((VL,), accum_dtype)
                    P_bf16_frag_0 = T.alloc_frag((VL,), dtype)
                    P_bf16_frag_1 = T.alloc_frag((VL,), dtype)
                    m_frag = T.alloc_frag((1,), accum_dtype)
                    m_new_frag = T.alloc_frag((1,), accum_dtype)
                    l_frag = T.alloc_frag((1,), accum_dtype)
                    alpha_frag = T.alloc_frag((1,), accum_dtype)
                    row_max_frag_0 = T.alloc_frag((1,), accum_dtype)
                    row_max_frag_1 = T.alloc_frag((1,), accum_dtype)
                    row_sum_frag_0 = T.alloc_frag((1,), accum_dtype)
                    row_sum_frag_1 = T.alloc_frag((1,), accum_dtype)
                    row_sum_frag = T.alloc_frag((1,), accum_dtype)
                    
                    for r in range(0, ROWS):
                        T.copy(S_ub[r, 0 : VL], S_frag_0)
                        T.copy(S_ub[r, VL :], S_frag_1)
                        T.copy(m_ub[r : r + 1], m_frag)
                        T.copy(l_ub[r : r + 1], l_frag)
                        T.vmuls(S_frag_0, scale_ln2, S_frag_0)
                        T.vmuls(S_frag_1, scale_ln2, S_frag_1)
                        # frag_0 & frag_1
                        T.vreduce_max(S_frag_0, row_max_frag_0)
                        T.vreduce_max(S_frag_1, row_max_frag_1)
                        # new_max = max(frag_0, frag_1, m_frag)
                        T.vmax(row_max_frag_0, m_frag, m_new_frag)
                        T.vmax(row_max_frag_1, m_new_frag, m_new_frag)
                        T.vexpdif(S_frag_0, m_new_frag, P_frag_0)
                        T.vexpdif(S_frag_1, m_new_frag, P_frag_1)
                        T.vcvt(P_frag_0, P_bf16_frag_0, dtype)
                        T.vcvt(P_frag_1, P_bf16_frag_1, dtype)

                        T.copy(P_bf16_frag_0, P_ub[r, 0 : VL])
                        T.copy(P_bf16_frag_1, P_ub[r, VL:])
                        T.vreduce_sum(P_frag_0, row_sum_frag_0)
                        T.vreduce_sum(P_frag_1, row_sum_frag_1)
                        T.vadd(row_sum_frag_0, row_sum_frag_1, row_sum_frag)

                        T.vexpdif(m_frag, m_new_frag, alpha_frag)
                        T.vmul(l_frag, alpha_frag, l_frag)
                        T.vadd(l_frag, row_sum_frag, l_frag)
                        T.copy(l_frag, l_ub[r: r + 1])
                        T.copy(m_new_frag, m_ub[r: r + 1])
                        T.copy(alpha_frag, alpha_ub[r: r + 1])

                T.copy(P_ub, P_shared, split_dim=-1)
                T.gemm(P_shared, V_shared, acc_o, transpose_B=True, clear_accum=True)
                T.copy(acc_o, O_tmp_ub, split_dim=-1)

                with T.SimdVF():
                    for i, j in T.Parallel(BR // 2, D):
                        O_ub[i, j] = O_ub[i, j] * alpha_ub[i] + O_tmp_ub[i, j]

            with T.SimdVF():
                for i, j in T.Parallel(BR // 2, D):
                    O_ub[i, j] = O_ub[i, j] / l_ub[i]

            T.copy(O_ub, O[bx * BR : (bx + 1) * BR, 0:D], split_dim=-1)

    return main

if __name__ == "__main__":
    program = flash_attention()
    tilelang.lower(program, target="tile")
