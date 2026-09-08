"""FlashAttention forward kernel on Ascend NPU using Mix kernel (Cube + Vector).

Online-softmax FlashAttention: O = softmax(Q @ K^T / sqrt(d)) @ V

UB buffers are full-shape (BR, BC) / (BR, D) tiles. The online softmax steps
(row-max, exp, sum, rescale, normalize) use whole-block fragment vector ops
(``vmul``/``vadd``/``vsub``/``vdiv``/``vexp``/``vreduce_*``) plus
``T.broadcast`` to expand per-row (BR,) vectors to (BR, N).
"""

import math

import tilelang
import tilelang.language as T


def flash_attention(D=128, SEQ_LEN=4096):
    BR = 64
    BC = 64
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
            V_shared = T.alloc_shared((BC, D), dtype)
            P_shared = T.alloc_shared((BR, BC), dtype)

            acc_s = T.alloc_fragment((BR, BC), accum_dtype)
            acc_o = T.alloc_fragment((BR, D), accum_dtype)

            S_ub = T.alloc_shared((BR, BC), accum_dtype)
            P_ub = T.alloc_shared((BR, BC), dtype)
            O_ub = T.alloc_shared((BR, D), accum_dtype)
            O_tmp_ub = T.alloc_shared((BR, D), accum_dtype)
            m_ub = T.alloc_shared((BR,), accum_dtype)
            old_m_ub = T.alloc_shared((BR,), accum_dtype)
            row_sum_ub = T.alloc_shared((BR,), accum_dtype)
            l_ub = T.alloc_shared((BR,), accum_dtype)
            alpha_ub = T.alloc_shared((BR,), accum_dtype)

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
                )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                T.copy(acc_s, S_ub)

                scale_ln2 = scale * 0.6931471805599453

                # --- Step 1: S = S * scale, m_new = row_max(S) ---
                with T.SimdVF():
                    S_frag = T.alloc_frag((BR, BC), accum_dtype)
                    row_max_frag = T.alloc_frag((BR,), accum_dtype)
                    T.copy(m_ub, old_m_ub)
                    T.copy(S_ub, S_frag)

                    # vmuls does not support rank > 1, so fill a full-shape
                    # vector with the scale and use vmul.
                    scale_vec = T.alloc_frag((BR, BC), accum_dtype)
                    T.fill(scale_vec, scale_ln2)
                    T.vmul(S_frag, scale_vec, S_frag)

                    T.copy(S_frag, S_ub)
                    T.vreduce_max(S_frag, row_max_frag)
                    T.copy(row_max_frag, m_ub)

                # --- Step 2: P = exp(S - m_new), row sums ---
                with T.SimdVF():
                    S_frag = T.alloc_frag((BR, BC), accum_dtype)
                    P_frag = T.alloc_frag((BR, BC), accum_dtype)
                    P_bf16_frag = T.alloc_frag((BR, BC), dtype)
                    m_frag = T.alloc_frag((BR,), accum_dtype)
                    m_new_frag = T.alloc_frag((BR,), accum_dtype)
                    row_max_frag = T.alloc_frag((BR,), accum_dtype)
                    row_sum_frag = T.alloc_frag((BR,), accum_dtype)

                    T.copy(old_m_ub, m_frag)
                    T.copy(m_ub, row_max_frag)
                    T.copy(S_ub, S_frag)

                    T.vmax(row_max_frag, m_frag, m_new_frag)
                    T.copy(m_new_frag, m_ub)

                    T.vsub(S_frag, m_new_frag, P_frag)
                    T.vexp(P_frag, P_frag)

                    T.vcvt(P_frag, P_bf16_frag, dtype)
                    T.copy(P_bf16_frag, P_ub)
                    T.vreduce_sum(P_frag, row_sum_frag)
                    T.copy(row_sum_frag, row_sum_ub)

                # --- Step 3: alpha = exp(m_old - m_new), l = l*alpha + row_sum ---
                with T.SimdVF():
                    m_frag = T.alloc_frag((BR,), accum_dtype)
                    m_new_frag = T.alloc_frag((BR,), accum_dtype)
                    l_frag = T.alloc_frag((BR,), accum_dtype)
                    alpha_frag = T.alloc_frag((BR,), accum_dtype)
                    row_sum_frag = T.alloc_frag((BR,), accum_dtype)

                    T.copy(old_m_ub, m_frag)
                    T.copy(m_ub, m_new_frag)
                    T.copy(l_ub, l_frag)
                    T.copy(row_sum_ub, row_sum_frag)

                    T.vsub(m_frag, m_new_frag, alpha_frag)
                    T.vexp(alpha_frag, alpha_frag)

                    T.copy(alpha_frag, alpha_ub)
                    T.vmul(l_frag, alpha_frag, l_frag)
                    T.vadd(l_frag, row_sum_frag, l_frag)
                    T.copy(l_frag, l_ub)

                # --- O = O * alpha + P @ V ---
                T.copy(P_ub, P_shared)
                T.gemm(P_shared, V_shared, acc_o, transpose_B=False, clear_accum=True)
                T.copy(acc_o, O_tmp_ub)

                with T.SimdVF():
                    O_frag = T.alloc_frag((BR, D), accum_dtype)
                    O_tmp_frag = T.alloc_frag((BR, D), accum_dtype)
                    alpha_frag = T.alloc_frag((BR,), accum_dtype)
                    alpha_bc = T.alloc_frag((BR, D), accum_dtype)
                    T.copy(O_ub, O_frag)
                    T.copy(O_tmp_ub, O_tmp_frag)
                    T.copy(alpha_ub, alpha_frag)
                    T.broadcast(alpha_frag, alpha_bc)
                    T.vmul(O_frag, alpha_bc, O_frag)
                    T.vadd(O_frag, O_tmp_frag, O_frag)
                    T.copy(O_frag, O_ub)

            # --- O /= l ---
            with T.SimdVF():
                O_frag = T.alloc_frag((BR, D), accum_dtype)
                l_frag = T.alloc_frag((BR,), accum_dtype)
                l_bc = T.alloc_frag((BR, D), accum_dtype)
                T.copy(O_ub, O_frag)
                T.copy(l_ub, l_frag)
                T.broadcast(l_frag, l_bc)
                T.vdiv(O_frag, l_bc, O_frag)
                T.copy(O_frag, O_ub)

            T.copy(O_ub, O[bx * BR : (bx + 1) * BR, 0:D])

    return main


if __name__ == "__main__":
    program = flash_attention()
    tilelang.lower(program, target="tile")
