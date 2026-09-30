#include <torch/library.h>
#include <torch/all.h>
void sm70_fp8_e5m2_cache_write(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,std::optional<torch::Tensor>,std::optional<torch::Tensor>);
TORCH_LIBRARY(sglang_sm70_turbomind, m) {
    m.def("fp8_e5m2_cache_write(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor locations, Tensor? k_scale, Tensor? v_scale) -> ()");
    m.impl("fp8_e5m2_cache_write", torch::kCUDA, &sm70_fp8_e5m2_cache_write);
}
