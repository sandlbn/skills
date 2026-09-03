# What sycl-tla actually has

The full example and benchmark inventory. Load this when `SKILL.md`'s transformer
mapping does not cover the shape, when the question is whether a kernel exists at
all, or when a target architecture has to be read off a name.

The example directory is the honest answer to what this library can do today. It
moves, so re-check it against the checked-out revision rather than trusting the
list below to stay complete.

## Reading the names

The prefix carries the target architecture, which is the only place it is
recorded consistently:

| Prefix | Architecture | Target |
|---|---|---|
| `bmg_`, `xe20_` | Xe2, Battlemage | `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31`, or `"bmg"` |
| `xe35_` | Xe3p | `intel_gpu_cri` |

Do not read architecture support out of `media/docs/cpp/functionality.md`. The
README links it for exactly this question, but the file came with the fork
unmodified and describes NVIDIA compute capabilities; it contains no Xe content.

## Examples

| Kernel shape | Example |
|---|---|
| Plain GEMM | `00_bmg_gemm` |
| GEMM via the collective builder | `01_bmg_gemm_with_collective_builder` |
| Mixed dtype (BF16/FP16 with INT8 or INT4), including dequantization | `02_bmg_gemm_mixed_dtype` |
| Stream-K scheduling, for load balance across uneven tiles | `03_bmg_gemm_streamk` |
| Grouped GEMM — a batch of distinct problem sizes | `04_bmg_grouped_gemm` |
| Fused epilogues, the EVT worked set | `05_bmg_gemm_with_epilogues` |
| Flash attention v2 — a family, see below | `06_bmg_flash_attention` |
| Two GEMMs sharing an A matrix, fused into one kernel | `07_bmg_dual_gemm` |
| FP8 inputs to FP32 output | `08_bmg_gemm_f8` |
| Grouped GEMM in FP8 | `09_bmg_grouped_gemm_f8` |
| Grouped GEMM in mixed dtype (`bf16_f16_s8`, `f16_u4`) | `10_bmg_grouped_gemm_mixed_dtype` |
| A BF16 GEMM exported from a shared library | `11_xe20_cutlass_library` |
| Mixture-of-experts GEMM on the CuTe grouped-GEMM interface | `12_xe20_moe_gemm_cute_interface` |
| GEMM with a bias epilogue | `13_bmg_gemm_bias` |
| Block-scaled GEMM (Xe3p) | `50_xe35_block_scaled_gemm` |
| Block-scaled grouped GEMM — E2M1, E4M3, E5M2 (Xe3p) | `51_xe35_block_scaled_grouped_gemm` |
| CuTe on its own, without CUTLASS | `cute/tutorial/` |
| A GEMM that runs on any SYCL device | `generics/device_agnostic/` |

## Attention

`06_bmg_flash_attention/` holds one file per variant, and the variant matters
more than the directory:

| Attention shape | File |
|---|---|
| Prefill | `06_xe_fmha_fwd.cpp` |
| Decode against a cached/paged KV | `06_xe_fmha_fwd_cached_kv.cpp` |
| Prefill also returning log-sum-exp | `06_xe_fmha_fwd_lse.cpp` |
| FP8 KV cache feeding FP16 MMA | `06_xe_fmha_fwd_fp8kvfp16mma.cpp` |
| Block-scaled MXFP | `06_xe_fmha_fwd_mxfp.cpp` |

The composable pieces are in `applications/flash_attention_v2/` — a mainloop, a
softmax epilogue, an SLM block copy, fusion hooks, and a tile scheduler. Build a
new attention variant from those rather than by editing an example in place.

Everything is `_fwd`. There is no backward kernel anywhere in the tree, so this
covers inference and serving, not training.

## Benchmarks

`benchmarks/` enumerates the configurations more completely than `examples/`
does, and upstream states plainly that the examples are *not* for benchmarking —
so a timing question belongs here. `benchmarks/flash_attention/` alone covers
prefill and decode across BF16, FP8, FP8-KV with FP16 MMA, and MXFP4/MXFP8;
`benchmarks/gemm/` and `benchmarks/grouped_gemm/` cover the matmul side.

## What is not here

- **Convolution.** `include/cutlass/conv/` and the implicit-GEMM convolution doc
  came with the fork and have no Xe implementation, example or benchmark. Use
  oneDNN.
- **Attention backward**, as above.
- **The command-line profiler.** `tools/profiler/` exists but is documented as
  not supported for SYCL.
