# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when a statement in the skill
needs to be checked against upstream, or when the answer depends on architecture
support, kernel coverage, or API surface that changes between revisions.

## sycl-tla

| Source | Use it for |
|---|---|
| [intel/sycl-tla](https://github.com/intel/sycl-tla) | entry point; the revision this skill's guidance is written against |
| [README](https://github.com/intel/sycl-tla/blob/main/README.md) | **the authoritative build answer.** Supported GPUs and their `DPCPP_SYCL_TARGET` values, required CMake flags, the minimum DPC++ version, and the "Validated Software Configurations" table — which is a different and stricter set of versions than the stated minimum |
| [examples/](https://github.com/intel/sycl-tla/tree/main/examples) | **the authoritative coverage answer.** Which kernel shapes exist is what the example directory holds — check here before telling a user a shape is unsupported. The `bmg_`/`xe20_` and `xe35_` name prefixes are how a target architecture is signalled |
| [benchmarks/](https://github.com/intel/sycl-tla/tree/main/benchmarks) | which *configurations* of a shape are exercised — the attention benchmarks enumerate the dtype and KV-cache variants more completely than the examples do. Upstream states the examples are not for benchmarking; this is the directory that is |
| [applications/](https://github.com/intel/sycl-tla/tree/main/applications) | the composable pieces behind the higher-level kernels — `flash_attention_v2/` (mainloop, softmax epilogue, tile scheduler) and `dual_gemm/`. Read here to build an attention variant rather than to call one |
| [Xe design docs](https://github.com/intel/sycl-tla/tree/main/media/docs/cpp) | the Intel-specific documents, which are the ones prefixed `xe_` — `xe_rearchitecture.md`, `xe_slm_pipeline.md`, `xe_bdpas_unified_block_scaled_mma.md` |
| [CHANGELOG](https://github.com/intel/sycl-tla/blob/main/CHANGELOG.md) | what landed in a revision, and which APIs were deprecated in it |
| [CuTe documentation](https://github.com/intel/sycl-tla/tree/main/media/docs) | layouts, tensors and atoms; the CUTLASS 3.x design the Intel port follows |

## Upstream CUTLASS

| Source | Use it for |
|---|---|
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | the API sycl-tla forked, when a CuTe concept is documented there and not yet in the Intel tree |

Read it for concepts, not for targets: the architecture tags, atoms and tile
shapes in that tree are NVIDIA's, and copying them into a sycl-tla kernel is the
most common way a port fails to build.

The same warning applies *inside* the Intel repository. It is a fork, so a page
sitting in `media/docs/` is not automatically about Intel — `functionality.md`
and `implicit_gemm_convolution.md` are unmodified NVIDIA documents describing
CUDA compute capabilities and kernels that have no Xe implementation, and the
sycl-tla README still links the first of them. Before trusting a doc page here,
check that it actually mentions Xe.

## Toolchain

| Source | Use it for |
|---|---|
| [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) | `icx` / `icpx` availability, versions, and the environment script |
| [Intel graphics driver releases](https://dgpu-docs.intel.com/) | the IGC and LTS driver versions a GPU needs at runtime |

## How to use these

- **Architecture questions are answered by the README, not from memory.** The
  `DPCPP_SYCL_TARGET` list grows as Intel ships new GPUs, and a value that looks
  plausible fails either at build time or, worse, at kernel launch.
- **The README names SKUs sparsely; do not read that as the support list.** It
  names one Arc part by model, while the targets are per die and cover the
  family. An architecture the README does not print for a target is not a fact
  to infer — leave it unstated rather than guessing it from a sibling row.
- **Coverage questions are answered by the example list.** A kernel shape exists
  if there is an example or a test for it; the absence of one is the real answer
  for the current revision, not a gap to work around silently.
- **Do not paraphrase a performance number out of these pages into an answer.**
  Numbers there describe someone else's hardware and someone else's problem
  shape. `SKILL.md` says how to measure; the user's own measurement is the only
  citable number.
- **Prefer the revision-matched page.** This library changes quickly and has
  deprecated APIs in flight, so ask which commit or release the user has checked
  out when a question turns on whether an API still exists.
- **The old name still resolves.** Sources filed under `cutlass-sycl` describe
  this project; treat them as current unless the content itself is stale.
