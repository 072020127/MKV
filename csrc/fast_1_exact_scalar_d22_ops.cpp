// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>

void makv_fast_1_exact_scalar_d22_chunk_out_cuda(
    const at::Tensor& q_logits,
    const at::Tensor& delta_logits,
    const at::Tensor& output_vhat,
    const at::Tensor& output_delta_v,
    const at::Tensor& baseline_output,
    const at::Tensor& value,
    const at::Tensor& reconstructed_value,
    const at::Tensor& row_max,
    const at::Tensor& normalizer,
    const at::Tensor& positions,
    const at::Tensor& probes,
    const at::Tensor& valid,
    const at::Tensor& probe_valid,
    at::Tensor& output);

namespace {

void fast_1_exact_scalar_d22_chunk_out(
    const at::Tensor& q_logits,
    const at::Tensor& delta_logits,
    const at::Tensor& output_vhat,
    const at::Tensor& output_delta_v,
    const at::Tensor& baseline_output,
    const at::Tensor& value,
    const at::Tensor& reconstructed_value,
    const at::Tensor& row_max,
    const at::Tensor& normalizer,
    const at::Tensor& positions,
    const at::Tensor& probes,
    const at::Tensor& valid,
    const at::Tensor& probe_valid,
    at::Tensor& output) {
  TORCH_CHECK(q_logits.is_cuda() && delta_logits.is_cuda() &&
                  output_vhat.is_cuda() && output_delta_v.is_cuda() &&
                  baseline_output.is_cuda() && value.is_cuda() &&
                  reconstructed_value.is_cuda() && row_max.is_cuda() &&
                  normalizer.is_cuda() && positions.is_cuda() &&
                  probes.is_cuda() && valid.is_cuda() &&
                  probe_valid.is_cuda() && output.is_cuda(),
              "exact scalar D22 inputs must be CUDA tensors");
  const auto device = q_logits.device();
  TORCH_CHECK(delta_logits.device() == device &&
                  output_vhat.device() == device &&
                  output_delta_v.device() == device &&
                  baseline_output.device() == device &&
                  value.device() == device &&
                  reconstructed_value.device() == device &&
                  row_max.device() == device && normalizer.device() == device &&
                  positions.device() == device && probes.device() == device &&
                  valid.device() == device && probe_valid.device() == device &&
                  output.device() == device,
              "exact scalar D22 inputs must use one CUDA device");
  TORCH_CHECK(q_logits.scalar_type() == at::kFloat &&
                  delta_logits.scalar_type() == at::kFloat &&
                  output_vhat.scalar_type() == at::kFloat &&
                  output_delta_v.scalar_type() == at::kFloat &&
                  baseline_output.scalar_type() == at::kFloat &&
                  row_max.scalar_type() == at::kFloat &&
                  normalizer.scalar_type() == at::kFloat &&
                  output.scalar_type() == at::kFloat,
              "exact scalar D22 GEMMs and accumulators must be float32");
  TORCH_CHECK(value.scalar_type() == at::kHalf ||
                  value.scalar_type() == at::kBFloat16 ||
                  value.scalar_type() == at::kFloat,
              "exact scalar D22 values must be float16, bfloat16, or float32");
  TORCH_CHECK(reconstructed_value.scalar_type() == value.scalar_type(),
              "exact scalar D22 value dtypes must match");
  TORCH_CHECK(positions.scalar_type() == at::kLong &&
                  probes.scalar_type() == at::kLong,
              "exact scalar D22 positions and probes must be int64");
  TORCH_CHECK(valid.scalar_type() == at::kBool &&
                  probe_valid.scalar_type() == at::kBool,
              "exact scalar D22 masks must be bool");
  TORCH_CHECK(q_logits.dim() == 4 && delta_logits.sizes() == q_logits.sizes() &&
                  output_vhat.sizes() == q_logits.sizes() &&
                  output_delta_v.sizes() == q_logits.sizes(),
              "exact scalar D22 GEMM shapes must match [KV,G,P,C]");
  const int64_t kv_heads = q_logits.size(0);
  const int64_t groups = q_logits.size(1);
  const int64_t probe_count = q_logits.size(2);
  const int64_t candidates = q_logits.size(3);
  TORCH_CHECK(kv_heads > 0 && groups > 0 && probe_count >= 0 && candidates >= 0,
              "exact scalar D22 dimensions must be non-negative");
  TORCH_CHECK(baseline_output.dim() == 3 &&
                  baseline_output.size(0) == kv_heads * groups &&
                  baseline_output.size(1) == probe_count,
              "exact scalar D22 baseline shape must be [KV*G,P,D]");
  TORCH_CHECK(value.dim() == 3 && reconstructed_value.sizes() == value.sizes() &&
                  value.size(0) == kv_heads && value.size(1) == candidates &&
                  value.size(2) == baseline_output.size(2),
              "exact scalar D22 value shape must be [KV,C,D]");
  TORCH_CHECK(row_max.dim() == 2 && row_max.size(0) == kv_heads * groups &&
                  row_max.size(1) == probe_count &&
                  normalizer.sizes() == row_max.sizes() &&
                  probe_valid.sizes() == row_max.sizes(),
              "exact scalar D22 baseline metadata shape mismatch");
  TORCH_CHECK(positions.dim() == 1 && probes.dim() == 1 && valid.dim() == 1 &&
                  output.dim() == 1 && positions.numel() == candidates &&
                  valid.numel() == candidates && probes.numel() == probe_count &&
                  output.numel() == candidates,
              "exact scalar D22 positions/output shape mismatch");
  TORCH_CHECK(q_logits.is_contiguous() && delta_logits.is_contiguous() &&
                  output_vhat.is_contiguous() && output_delta_v.is_contiguous() &&
                  baseline_output.is_contiguous() && row_max.is_contiguous() &&
                  normalizer.is_contiguous() && positions.is_contiguous() &&
                  probes.is_contiguous() && valid.is_contiguous() &&
                  probe_valid.is_contiguous() && output.is_contiguous(),
              "exact scalar D22 metadata and GEMM tensors must be contiguous");
  TORCH_CHECK(value.stride(2) == 1 && reconstructed_value.stride(2) == 1,
              "exact scalar D22 values must be contiguous in head dimension");
  c10::cuda::CUDAGuard device_guard(device);
  makv_fast_1_exact_scalar_d22_chunk_out_cuda(
      q_logits, delta_logits, output_vhat, output_delta_v, baseline_output,
      value, reconstructed_value, row_max, normalizer, positions, probes, valid,
      probe_valid, output);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(lmcache_makv, m) {
  m.def(
      "fast_1_exact_scalar_d22_chunk_out(Tensor q_logits, "
      "Tensor delta_logits, Tensor output_vhat, Tensor output_delta_v, "
      "Tensor baseline_output, Tensor value, Tensor reconstructed_value, "
      "Tensor row_max, Tensor normalizer, Tensor positions, Tensor probes, "
      "Tensor valid, Tensor probe_valid, Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(lmcache_makv, CUDA, m) {
  m.impl("fast_1_exact_scalar_d22_chunk_out",
         &fast_1_exact_scalar_d22_chunk_out);
}
