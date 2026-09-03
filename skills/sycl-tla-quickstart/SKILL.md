---
name: sycl-tla-quickstart
description: >-
  Writing and building GEMM, attention and fused-epilogue kernels for Intel GPUs
  with sycl-tla, the SYCL CUTLASS/CuTe implementation for Xe (formerly
  CUTLASS-SYCL). Use when a matmul or attention kernel has to be written by hand
  rather than called through a library, when a CUTLASS or CuTe kernel has to move
  to Intel Arc or Intel Data Center GPU Crescent Island, when a build needs the
  right DPCPP_SYCL_TARGET for the architecture, when an elementwise operation
  should be fused into a GEMM epilogue instead of costing a second kernel launch,
  or when the shape of the problem is flash attention prefill or cached-KV decode,
  mixture-of-experts, grouped GEMM, stream-K, block-scaled MXFP, or FP8 and INT4
  mixed-dtype matmul. Covers the oneAPI and icpx build, the architecture target
  table, the example that matches a kernel shape, and the Epilogue Visitor Tree.
license: Apache-2.0
compatibility: "Requires oneAPI DPC++ 2025.1 or newer (icx/icpx), CMake, Ninja, and C++17; upstream CI validates on DPC++ 2025.3+ with G++ 13 on Ubuntu 25.04. Header-only, so nothing links; running a kernel needs an Intel Arc GPU (Xe2) or Intel Data Center GPU Crescent Island (Xe3p)."
metadata:
  intel-skill-type: "tool-skill"
  version: "1.0"
---

# sycl-tla quickstart

## Purpose

Builds hand-written kernels for Intel Xe GPUs with sycl-tla — a fork of NVIDIA
CUTLASS that extends the CUTLASS and CuTe APIs to Intel hardware through SYCL.
The library is header-only C++ templates: hierarchical tiling, composable
policies, and data-movement primitives that target the Xe systolic units (DPAS).

GEMM is the foundation but not the boundary. Fused multi-head attention is a
first-class kernel family here, with its own mainloop, softmax epilogue and tile
scheduler under `applications/flash_attention_v2/`, covering prefill and
cached-KV decode across BF16, FP8, FP8-KV with FP16 MMA, and MXFP4/MXFP8. Above
those sit mixture-of-experts and grouped GEMM, dual GEMM, and block-scaled matmul.

Reach for sycl-tla when a kernel has to be *written*, not called. A GEMM that a
library already covers belongs in that library; sycl-tla earns its complexity
when the kernel needs a shape, a data type, or an epilogue fusion that no library
exposes — and the fusion is usually the reason, because an operation folded into
the epilogue runs on accumulator data still in registers instead of paying
another kernel launch and another round trip to global memory.

## When to Use This Skill

Use this skill when:

- A GEMM, attention or fused-epilogue kernel is being written by hand for an
  Intel GPU.
- A CUTLASS or CuTe kernel written for NVIDIA has to run on Intel Xe.
- A build fails because the SYCL target or the CUTLASS SYCL switch is wrong.
- An elementwise op (bias, ReLU, GELU, scaling) should be fused into a GEMM.
- The kernel shape is flash attention — prefill or cached-KV decode, including
  quantized KV — mixture-of-experts, grouped GEMM, dual GEMM, stream-K, or a
  mixed-precision or block-scaled matmul.

Do **not** use this skill for array-level math that dpnp or oneMKL already
covers, for writing Triton kernels for Intel GPUs, for serving or benchmarking a
model, or for profiling a kernel once it builds and runs.

## Quick Start

sycl-tla is header-only, so this builds the shipped examples and unit tests —
there is no library to install afterwards.

```bash
source /opt/intel/oneapi/setvars.sh

git clone https://github.com/intel/sycl-tla
mkdir sycl-tla/build && cd sycl-tla/build

CC=icx CXX=icpx cmake .. -G Ninja \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21"

ninja test_unit -j$(nproc)
```

Both `-DCUTLASS_ENABLE_SYCL=ON` and `CC=icx CXX=icpx` are required. Without the
first, CMake configures the NVIDIA code paths; without the second it picks up the
system compiler, which cannot compile SYCL.

## Implementation Guide

1. **Set `DPCPP_SYCL_TARGET` to the architecture, not the product name.** This is
   the flag that most often makes a working source tree fail to build, and a
   wrong value produces a binary that will not run on the machine that built it:

   | `DPCPP_SYCL_TARGET` | GPU | Architecture |
   |---|---|---|
   | `intel_gpu_bmg_g21` | Battlemage G21 die — the Arc B580 upstream names by SKU | Xe2 |
   | `intel_gpu_bmg_g31` | Battlemage G31 die | Xe2 |
   | `intel_gpu_cri` | Intel Data Center GPU Crescent Island | Xe3p |
   | `"bmg"` | both Battlemage dies, for one binary across the family | Xe2 |

   Xe2 (Battlemage) is a family, not a part, and the family is supported: the
   Arc B-series desktop cards (B570, B580) and the Arc Pro B-series (B50, B60,
   B70) all run these kernels. The flag follows the **die**, not the retail
   name, so it is `g21` or `g31` that decides it — a higher model number does
   not imply the larger die.

   Upstream's README names only the B580, so when the die for a part is not
   known, build with `"bmg"` and get both rather than guessing one. Which
   kernels a target supports is read off the example and benchmark names —
   `bmg_`/`xe20_` prefixes are Xe2, `xe35_` is Xe3p — not off
   `media/docs/cpp/functionality.md`, which the README points at but which is
   still the inherited NVIDIA document.

2. **Start from the example whose shape matches, not from a blank file.** The
   examples under `examples/` in the upstream repository are the intended entry
   point, and they are named for the kernel they build. For a transformer, which
   is what this library is aimed at, the layer maps onto them like this:

   | Transformer piece | Starts from |
   |---|---|
   | QKV / output projection, MLP up and down | `00`–`01`, or `13_bmg_gemm_bias` with the bias folded in |
   | Attention, prefill | `06_bmg_flash_attention` → `06_xe_fmha_fwd.cpp` |
   | Attention, decode against a KV cache | `06_bmg_flash_attention` → `06_xe_fmha_fwd_cached_kv.cpp` |
   | SwiGLU / gated MLP — two projections over one input | `07_bmg_dual_gemm` |
   | MoE expert dispatch | `12_xe20_moe_gemm_cute_interface`, or `04_bmg_grouped_gemm` |
   | Weight-only quantized linear (INT4/INT8 weights) | `02_bmg_gemm_mixed_dtype` |
   | FP8 or MXFP4/MXFP8 quantized path | `08_bmg_gemm_f8`, `50_xe35_block_scaled_gemm` |
   | Norm, residual, activation next to a matmul | not a kernel — fold into the epilogue, step 3 |

   For the full example list, and the stream-K, dual-GEMM and block-scaled
   entries this table leaves out, load
   [`references/kernel-map.md`](references/kernel-map.md).

   The attention example is itself a family rather than one file, and picking the
   right variant inside it matters more than picking the directory:

   | Attention shape | File in `06_bmg_flash_attention/` |
   |---|---|
   | Prefill — full Q against full KV | `06_xe_fmha_fwd.cpp` |
   | Decode — Q against a cached/paged KV | `06_xe_fmha_fwd_cached_kv.cpp` |
   | Prefill also returning log-sum-exp, for split-KV or chunked softmax merging | `06_xe_fmha_fwd_lse.cpp` |
   | Quantized KV cache — FP8 KV feeding FP16 MMA | `06_xe_fmha_fwd_fp8kvfp16mma.cpp` |
   | Block-scaled MXFP attention | `06_xe_fmha_fwd_mxfp.cpp` |

   The pieces those build from live in `applications/flash_attention_v2/`
   (`xe_fmha_fwd_mainloop.hpp`, `xe_fmha_fwd_epilogue.hpp`, `xe_tile_scheduler.hpp`),
   which is where a custom attention variant is composed. Only **forward** exists;
   there is no attention backward kernel, so training a fused attention on Intel
   through this library is not on the table today.

3. **Fuse the epilogue with an Epilogue Visitor Tree.** EVT composes the
   epilogue from nodes — the accumulator, operands loaded per row or column, and
   elementwise ops over them — so bias, activation, scaling and residual add
   become part of the GEMM rather than separate kernels. `05_bmg_gemm_with_epilogues`
   is the worked set; build the tree there before writing one from scratch.

4. **Pick the data type from what the epilogue needs, not only the inputs.**
   sycl-tla covers FP64, FP32, FP16 and BF16, the FP8 types E5M2 and E4M3, and
   4- and 8-bit signed and unsigned integers, with tensor-wise, channel-wise and
   group-wise quantization including zero points. A quantized GEMM usually
   accumulates in INT32 and dequantizes in the epilogue, so the scale operands
   are part of the epilogue design.

5. **Consume it as headers.** There is nothing to link: add the upstream
   `include/` directory to the include path of the project that uses it, keep
   `-DCUTLASS_ENABLE_SYCL` defined in that project too, and compile with `icpx`.
   Projects that vendor it commonly pull it in with CMake `FetchContent`.

6. **Verify before tuning.** `ninja test_unit` runs the Google Test suite over
   the core API and full GEMM computations. A kernel that fails a correctness
   check is not a kernel worth timing, and tile-size or scheduling changes are
   the easiest way to break one silently.

## Performance

No measured numbers ship with this skill, and a hand-written kernel is not
automatically faster than the library call it replaces. What to measure:

- Compare against the library path the kernel is meant to beat, at the real
  problem shape, not a square one.
- Time the fused kernel against the unfused sequence it replaces. Fusion pays
  when the epilogue is memory-bound; when it is not, it mostly adds register
  pressure.
- Watch register pressure and spills. A tile that is too large spills, and a
  spilling kernel can be slower than a smaller tile that does not.
- Measure with the same `DPCPP_SYCL_TARGET` the deployment uses. A binary built
  for one architecture says nothing about another.
- Vary the tile shape. It is the first tuning knob and usually the largest.

## Gotchas & Limitations

- **`media/docs/cpp/functionality.md` is not about Intel.** The README links it
  for "which kernels require which target architectures", but the file is the
  inherited NVIDIA document — compute capabilities, CUDA toolkit versions, and
  unit-test links into `NVIDIA/cutlass`. It contains no Xe content at all, so an
  answer sourced from it will be confidently wrong. Read coverage off
  `examples/`, `benchmarks/` and `test/unit/` instead.
- **Attention is forward-only.** Every FMHA kernel and example in the tree is
  `_fwd`; there is no backward pass, so this library covers inference and
  prefill/decode serving, not training a fused attention.
- **The command-line profiler does not support SYCL yet.** `tools/profiler/`
  exists because the tree is a CUTLASS fork, but it is documented as not
  available for SYCL — profile the kernel with an Intel GPU profiler instead.
- **Legacy CuTe atom APIs are deprecated** and slated for removal. Code copied
  from an older example or from an NVIDIA CUTLASS tree may compile today and
  break at the next update; write against the current CuTe atom APIs.
- **The minimum compiler and the validated compiler are different numbers.**
  Upstream states a minimum of oneAPI 2025.1, but its CI validates Xe2 on DPC++
  2025.3 or newer with G++ 13 on Ubuntu 25.04, against a specific compute
  runtime and graphics compiler pair. Between the two lies a range that is
  allowed but untested — build failures there are worth checking against the
  validated configuration before being treated as a bug.
- **The graphics runtime has a floor too**, and a driver older than the compiler
  expects fails at kernel launch, not at build time. A clean build is not
  evidence the machine can run the kernel; upstream pins a compute-runtime and
  graphics-compiler pair per architecture in its validated-configurations table.
- **`-DDPCPP_HOST_COMPILER=g++-13` has no fallback.** If it is set, `g++-13`
  must resolve in `PATH` or the build fails.
- **This is not a drop-in port of an NVIDIA CUTLASS kernel.** The APIs are shared
  and the tiling model is the same, but the atoms, the tile shapes and the
  architecture tags differ, so a kernel is retargeted rather than recompiled.
- **There is no convolution for Xe.** `include/cutlass/conv/` and
  `implicit_gemm_convolution.md` come with the fork, but no Xe conv kernel,
  example or benchmark exists — convolution on Intel GPUs belongs in oneDNN.
  This library is aimed at transformer shapes.
- Not covered: writing a new CuTe atom, and multi-GPU or collective work above
  the kernel.

## References

| File | Load it when |
|---|---|
| [`references/kernel-map.md`](references/kernel-map.md) | the shape is not in the transformer mapping above, or the question is whether a kernel exists at all — the full example, attention-variant and benchmark inventory, and what the tree does *not* have |
| [`references/official-sources.md`](references/official-sources.md) | you need the architecture target for a GPU this skill does not list, the current CuTe atom API, which data types and quantization modes the installed revision supports, or upstream's own words on a claim here |

Two things here should not be answered from memory: **the `DPCPP_SYCL_TARGET`
value for a given GPU** (new architectures are added, and a plausible-looking
value fails at build or at launch) and **which kernel shapes and data types the
current revision supports** — this library moves quickly, and the example list is
the honest answer to what it can do today.
