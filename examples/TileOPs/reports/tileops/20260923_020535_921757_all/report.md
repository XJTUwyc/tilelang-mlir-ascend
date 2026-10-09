# TileOPs Evaluation Report: all

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-09-23T02:26:27.192219+00:00 |
| evaluation_scope | all |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20260923_020535_921757_all |
| git_commit | 54a19e28dfd8756daa253b387e00d10ba7f952e7 |
| correctness_target | tests/ops |
| benchmark_target | benchmarks/ops |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B2C x 16 |
| cpu | x86_64 |
| cann | 8.5.0 |
| driver | 26.0.rc1 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1 |
| tilelang | 0.1.2+ubuntu.22.4.npuir |
| python | 3.11.15 |
| os | Linux-5.15.0-191-generic-x86_64-with-glibc2.35 |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 6 | 70 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AdaLayerNormFwdOp | 100.0% (22/22) | 1.13e-02 | 8 | 0.43% – 67.15% |
| LerpTensorFwdOp | 100.0% (8/8) | 5.21e-08 | 10 | 47.88% – 77.63% |
| LogSumExpFwdOp | 100.0% (23/23) | 3.67e-08 | 6 | 3.18% – 36.78% |
| MishFwdOp | 100.0% (7/7) | 2.45e-05 | 9 | 27.19% – 64.03% |
| MultiHeadAttentionFwdOp | 100.0% (6/6) | 8.54e-04 | 10 | 2.66% – 23.39% |
| SSDChunkScanFwdOp | 100.0% (4/4) | 2.19e-04 | 11 | 0.13% – 20.68% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AdaLayerNormFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-dit | 3.2000 | 15.36 | {"m": 64, "n": 1152} | float32 | msprof | main | 1.8000 |
| smoke-dit | 3.3000 | 7.45 | {"m": 64, "n": 1152} | float16 | msprof | main | 1.8000 |
| smoke-dit | 3.2800 | 7.49 | {"m": 64, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| dit-xl-2 | 9.2800 | 42.04 | {"m": 1024, "n": 1152} | float16 | msprof | main | 1.8000 |
| dit-xl-2 | 9.1000 | 42.87 | {"m": 1024, "n": 1152} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 41.7600 | 66.96 | {"m": 2048, "n": 4096} | float16 | msprof | main | 1.8000 |
| llama-3.1-8b-prefill | 41.6400 | 67.15 | {"m": 2048, "n": 4096} | bfloat16 | msprof | main | 1.8000 |
| llama-3.1-8b-decode | 3.2400 | 0.43 | {"m": 1, "n": 4096} | bfloat16 | msprof | main | 1.8000 |

### LerpTensorFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 11.0200 | 63.44 | [1024, 1024] | float32 | msprof | main | 1.8000 |
| smoke-1m | 7.3000 | 47.88 | [1024, 1024] | float16 | msprof | main | 1.8000 |
| smoke-1m | 7.1800 | 48.68 | [1024, 1024] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 72.0400 | 77.63 | [4096, 4096] | float16 | msprof | main | 1.8000 |
| elementwise-16m | 72.7400 | 76.88 | [4096, 4096] | bfloat16 | msprof | main | 1.8000 |
| elementwise-16m | 156.1600 | 76.68 | [4096, 4096] | float32 | msprof | main | 1.8000 |
| elementwise-64m | 384.0800 | 69.84 | [8192, 8192] | float16 | msprof | main | 1.8000 |
| elementwise-64m | 380.6400 | 70.22 | [8192, 8192] | bfloat16 | msprof | main | 1.8000 |
| elementwise-256m | 1715.2000 | 67.90 | [16384, 16384] | float16 | msprof | main | 1.8000 |
| elementwise-256m | 1739.0800 | 66.98 | [16384, 16384] | bfloat16 | msprof | main | 1.8000 |

### LogSumExpFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| attn-weights-4k | 16.1400 | 28.87 | [32, 32, 4096] | float16 | msprof | main | 1.8000 |
| attn-weights-4k | 17.0600 | 27.41 | [32, 32, 4096] | bfloat16 | msprof | main | 1.8000 |
| attn-weights-32k | 101.3600 | 36.78 | [32, 32, 32768] | bfloat16 | msprof | main | 1.8000 |
| lm-head-logits | 14.3200 | 3.18 | [4, 102400] | float16 | msprof | main | 1.8000 |
| lm-head-logits | 13.9800 | 3.26 | [4, 102400] | bfloat16 | msprof | main | 1.8000 |
| 3d-multidim-reduce | 9.6800 | 24.07 | [4, 128, 4096] | float16 | msprof | main | 1.8000 |

### MishFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| smoke-1m | 8.7200 | 28.35 | [1048576] | float32 | msprof | main | 1.8000 |
| smoke-1m | 7.0000 | 27.19 | [1048576] | float16 | msprof | main | 1.8000 |
| smoke-1m | 7.1800 | 29.15 | [1048576] | bfloat16 | msprof | main | 1.8000 |
| yolo-p3 | 75.8200 | 62.72 | [16, 256, 80, 80] | float16 | msprof | main | 1.8000 |
| yolo-p3 | 81.6800 | 64.03 | [16, 256, 80, 80] | bfloat16 | msprof | main | 1.8000 |
| yolo-p4 | 41.3400 | 57.52 | [16, 512, 40, 40] | float16 | msprof | main | 1.8000 |
| yolo-p4 | 41.9000 | 62.41 | [16, 512, 40, 40] | bfloat16 | msprof | main | 1.8000 |
| fc-wide | 27.3200 | 55.71 | [2048, 4096] | float16 | msprof | main | 1.8000 |
| fc-wide | 27.9400 | 59.90 | [2048, 4096] | bfloat16 | msprof | main | 1.8000 |

### MultiHeadAttentionFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mha-fwd-smoke-s512-h8-d64 | 43.5600 | 2.68 | {"batch": 1, "causal": true, "dim": 64, "heads": 8, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| mha-fwd-smoke-s512-h8-d64 | 44.0400 | 2.66 | {"batch": 1, "causal": true, "dim": 64, "heads": 8, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 289.3400 | 14.55 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-short | 292.0600 | 14.42 | {"batch": 4, "causal": true, "dim": 128, "heads": 32, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 983.7400 | 23.39 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-8b-long | 1029.1000 | 22.38 | {"batch": 2, "causal": true, "dim": 128, "heads": 32, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 282.8400 | 14.88 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-short | 292.0600 | 14.42 | {"batch": 2, "causal": true, "dim": 128, "heads": 64, "seq_len": 512} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1001.5200 | 22.98 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | float16 | msprof | _gqa_prefill_fwd_main | 1.8000 |
| llama-3.1-70b-long | 1027.5400 | 22.41 | {"batch": 1, "causal": true, "dim": 128, "heads": 64, "seq_len": 2048} | bfloat16 | msprof | _gqa_prefill_fwd_main | 1.8000 |

### SSDChunkScanFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| b1-c2-L64-h4-p64-n32-fp16 | 31.5800 | 0.13 | {"batch": 1, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | float16 | msprof | main | 1.8000 |
| b2-c4-L64-h8-p64-n64-fp16 | 34.7600 | 1.33 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 64, "n_groups": 2, "n_heads": 8, "num_chunks": 4} | float16 | msprof | main | 1.8000 |
| b1-c2-L128-h4-p128-n32-bf16 | 35.7600 | 0.28 | {"batch": 1, "chunk_len": 128, "d_head": 128, "d_state": 32, "n_groups": 1, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| b2-c2-L64-h4-p64-n32-bf16 | 32.6400 | 0.30 | {"batch": 2, "chunk_len": 64, "d_head": 64, "d_state": 32, "n_groups": 2, "n_heads": 4, "num_chunks": 2} | bfloat16 | msprof | main | 1.8000 |
| latency-130m-4k | 113.0200 | 9.30 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-130m-4k | 735.3000 | 14.17 | {"batch": 8, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-130m-32k | 3012.8401 | 19.58 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 24, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| latency-2p7b-4k | 301.3200 | 11.63 | {"batch": 1, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| serving-2p7b-4k | 1173.5000 | 17.17 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 16} | float16 | msprof | main | 1.8000 |
| longctx-2p7b-32k | 4771.6401 | 20.68 | {"batch": 2, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 128} | float16 | msprof | main | 1.8000 |
| throughput-2p7b-2k | 581.4600 | 12.15 | {"batch": 4, "chunk_len": 256, "d_head": 64, "d_state": 128, "n_groups": 1, "n_heads": 80, "num_chunks": 8} | float16 | msprof | main | 1.8000 |
