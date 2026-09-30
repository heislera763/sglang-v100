# CUDA / Torch support findings

Do not treat the working Torch 2.9.1 + CUDA 12.8 installation as the final
V100-capable PyTorch release. CUDA toolkit support, PyTorch wheel architectures,
and inference-kernel support are separate constraints.

| Candidate | Volta support evidence | Local execution |
| --- | --- | --- |
| Torch 2.9.1 + cu128 | Existing wheel includes sm_70 | CUDA arithmetic, FP16 GEMM and Triton 3.5.1 pass |
| Torch 2.10 + cu128/cu129 | Last release before 2.11 removes Volta from these wheel variants | Not tested |
| Torch 2.13 + cu126 | Official wheel includes sm_70; selected to match mainline's Torch pin | CUDA arithmetic, FP16 GEMM and Triton 3.7.1 pass |
| Torch 2.14 + cu126 | Officially the last release with a prebuilt Volta wheel | Not tested |
| Torch 2.15+ standard wheels | cu126 removed; standard wheels drop Volta | Requires investigating a source build |
| CUDA toolkit 12.9 | Local nvcc lists compute_70 and sm_70 | SM70 AOT library and NVFP4 Marlin build pass; native import, E5M2, dense, MoE and QSA checks pass |
| CUDA toolkit 13.x | Offline compilation and library support before Turing removed | Unsuitable for the SM70 extension build |

PyTorch's official [2.11 packaging notice](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
removes Volta from CUDA 12.8 and 12.9 wheels while retaining it in cu126.
Its [2.15 packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
identifies 2.14 as the last release providing a prebuilt wheel for Volta.
The [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
record the removal of older architecture compilation and library support.

The profile uses the cu126 **runtime packaged with Torch** and the locally
installed CUDA 12.9 **compiler**. Those need not be the same minor release,
but actual compilation and GPU execution must pass. Triton 3.7.1 also passed
with its bundled CUDA 12.8.93 assembler; replacing it is not required for SM70.
The serving profile selects CUDA 12.9's ptxas to use the same local toolchain. The driver
version reported by nvidia-smi is not the CUDA toolkit or Torch runtime version.

Excluded audio and CUDA 13 acceleration packages are outside this Qwen
text/image serving profile. The profile is not a promise of support for every
upstream model or optional backend. Re-evaluate the dependency exclusions and
native build with each upstream update.
