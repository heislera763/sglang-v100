cmake_minimum_required(VERSION 3.26)
project(sgl-kernel-v100 LANGUAGES CXX CUDA)
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(FetchContent)
# Header-only dependencies: do not configure their own CUDA build targets.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(SGLANG_AOT_UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/../../python/sglang/kernels/aot" ABSOLUTE)
FetchContent_Declare(repo-cutlass URL https://github.com/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz URL_HASH SHA256=09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5)
FetchContent_Populate(repo-cutlass)
FetchContent_Declare(repo-flashinfer URL https://github.com/flashinfer-ai/flashinfer/archive/bc29697ba20b7e6bdb728ded98f04788e16ee021.tar.gz URL_HASH SHA256=931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d)
FetchContent_Populate(repo-flashinfer)
set(SOURCES
    "${SGLANG_AOT_UPSTREAM}/csrc/allreduce/custom_all_reduce.cu"
    "csrc/attention/cutlass_mla_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/attention/merge_attn_states.cu"
    "csrc/attention/vertical_slash_index.cu"
    "csrc/common_extension.cc"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/activation.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/concat_mla.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/copy.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/fused_add_rms_norm_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/pos_enc.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/elementwise/topk.cu"

    "csrc/gemm/bmm_fp8.cu"
    "csrc/gemm/dsv3_fused_a_gemm.cu"
    "csrc/gemm/dsv3_router_gemm_bf16_out.cu"
    "csrc/gemm/dsv3_router_gemm_entry.cu"
    "csrc/gemm/dsv3_router_gemm_float_out.cu"
    "csrc/gemm/fp8_blockwise_gemm_kernel.cu"
    "csrc/gemm/fp8_gemm_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/gemm/int8_gemm_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/gemm/per_token_group_quant_8bit.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/gemm/per_token_group_quant_8bit_v2.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/gemm/per_token_quant_fp8.cu"
    "csrc/gemm/qserve_w4a8_per_chn_gemm.cu"
    "csrc/gemm/qserve_w4a8_per_group_gemm.cu"
    "csrc/gemm/gptq/gptq_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/grammar/apply_token_bitmask_inplace_cuda.cu"

    "csrc/kvcacheio/transfer.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/mamba/causal_conv1d.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/memory/weak_ref_tensor.cpp"

    "${SGLANG_AOT_UPSTREAM}/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu"
    "csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu"
    "csrc/moe/moe_align_kernel.cu"
    "csrc/moe/moe_fused_gate.cu"
    "csrc/moe/kimi_k2_moe_fused_gate.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/moe/moe_sum.cu"
    "csrc/moe/moe_topk_softmax_kernels.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/moe/moe_topk_sigmoid_kernels.cu"
    "csrc/moe/fp8_blockwise_moe_kernel.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/moe/prepare_moe_input.cu"

    "${SGLANG_AOT_UPSTREAM}/csrc/quantization/gguf/gguf_kernel.cu"
    "csrc/speculative/eagle_utils.cu"
    "csrc/speculative/ngram_utils.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/speculative/packbit.cu"
    "${SGLANG_AOT_UPSTREAM}/csrc/speculative/speculative_sampling.cu"

    "${repo-flashinfer_SOURCE_DIR}/csrc/norm.cu"
    "${repo-flashinfer_SOURCE_DIR}/csrc/renorm.cu"
)

# Alternative sources still include unchanged sibling headers from upstream.
foreach(source IN LISTS SOURCES)
  if(NOT IS_ABSOLUTE "${source}")
    get_filename_component(source_dir "${source}" DIRECTORY)
    set_property(SOURCE "${source}" APPEND PROPERTY INCLUDE_DIRECTORIES "${SGLANG_AOT_UPSTREAM}/${source_dir}")
  endif()
endforeach()
# Mainline shares the sampling implementation with its JIT kernels.
set_property(SOURCE "${SGLANG_AOT_UPSTREAM}/csrc/speculative/speculative_sampling.cu"
  APPEND PROPERTY INCLUDE_DIRECTORIES "${SGLANG_AOT_UPSTREAM}/../jit/include")

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_include_directories(common_ops PRIVATE include csrc ${SGLANG_AOT_UPSTREAM}/include ${SGLANG_AOT_UPSTREAM}/csrc ${repo-cutlass_SOURCE_DIR}/include ${repo-cutlass_SOURCE_DIR}/tools/util/include ${repo-cutlass_SOURCE_DIR}/examples/common ${repo-flashinfer_SOURCE_DIR}/include ${repo-flashinfer_SOURCE_DIR}/csrc)
target_compile_definitions(common_ops PRIVATE SGL_KERNEL_V100_ONLY FLASHINFER_ENABLE_F16 ENABLE_NVFP4=1 CUTE_USE_PACKED_TUPLE=1)
target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;--threads=1> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} CUDA::cuda_driver CUDA::cudart CUDA::cublas CUDA::cublasLt)
install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel/sm70)
