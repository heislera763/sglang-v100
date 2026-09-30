/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");

  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");

  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");

  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");

  m.impl("gemma_fused_add_rmsnorm", torch::kCUDA, &gemma_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");

  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");

  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  m.def("concat_mla_k(Tensor! k, Tensor k_nope, Tensor k_rope) -> ()");

  m.impl("concat_mla_k", torch::kCUDA, &concat_mla_k);

  m.def("concat_mla_absorb_q(Tensor a, Tensor b, Tensor! out) -> ()");

  m.impl("concat_mla_absorb_q", torch::kCUDA, &concat_mla_absorb_q);

  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids, bool ignore_invalid_expert) -> ()");

  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, float "
      "moe_softcapping, Tensor? correction_bias) -> ()");

  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, Tensor? "
      "correction_bias) -> ()");

  m.impl("topk_sigmoid", torch::kCUDA, &topk_sigmoid);

  m.def("moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()");

  m.impl("moe_sum_reduce", torch::kCUDA, &moe_sum_reduce);

  m.def("moe_sum(Tensor input, Tensor! output) -> ()");

  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  m.def("weak_ref_tensor(Tensor tensor) -> Tensor");

  m.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  m.def("top_k_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_k_arr, int top_k_val) -> ()");

  m.impl("top_k_renorm_probs", torch::kCUDA, &top_k_renorm_probs);

  m.def("top_p_renorm_probs(Tensor probs, Tensor! renorm_probs, Tensor? maybe_top_p_arr, float top_p_val) -> ()");

  m.impl("top_p_renorm_probs", torch::kCUDA, &top_p_renorm_probs);

  m.def("apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()");

  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);

  m.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");

  m.impl("causal_conv1d_update", torch::kCUDA, &causal_conv1d_update);

  m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");

  m.impl("causal_conv1d_fwd", torch::kCUDA, &causal_conv1d_fwd);
  m.def(
      "tree_speculative_sampling_target_only(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor uniform_samples, Tensor uniform_samples_for_final_sampling, Tensor target_probs, Tensor draft_probs, "
      "float threshold_single, float threshold_acc, "
      "bool deterministic) -> ()");
  m.impl("tree_speculative_sampling_target_only", torch::kCUDA, &tree_speculative_sampling_target_only);
  m.def(
      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
      "Tensor target_predict) -> ()");
  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);
  m.def(
      "build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
      "Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, Tensor! retrive_next_token, "
      "Tensor! retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> "
      "()");
  m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient);
  m.def(
      "segment_packbits(Tensor x, Tensor input_indptr, Tensor output_indptr, Tensor! y, int batch_size, "
      "int cuda_stream) -> ()");
  m.impl("segment_packbits", torch::kCUDA, &segment_packbits);
}

REGISTER_EXTENSION(common_ops)
