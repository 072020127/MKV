// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>

void makv_fast_d22_chunk_out_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& row_max,
    const at::Tensor& normalizer,
    const at::Tensor& baseline_output,
    const at::Tensor& positions,
    const at::Tensor& probes,
    const at::Tensor& valid,
    const at::Tensor& probe_valid,
    at::Tensor& output,
    int64_t scale_dtype);

namespace {

void fast_d22_chunk_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& row_max,
    const at::Tensor& normalizer,
    const at::Tensor& baseline_output,
    const at::Tensor& positions,
    const at::Tensor& probes,
    const at::Tensor& valid,
    const at::Tensor& probe_valid,
    at::Tensor& output,
    int64_t scale_dtype) {
  TORCH_CHECK(scale_dtype == 0 || scale_dtype == 1,
              "fast D22 scale_dtype must be 0 (float16) or 1 (float32)");
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "fast D22 Q/K/V must be CUDA tensors");
  TORCH_CHECK(row_max.is_cuda() && normalizer.is_cuda() &&
                  baseline_output.is_cuda() && positions.is_cuda() &&
                  probes.is_cuda() && valid.is_cuda() &&
                  probe_valid.is_cuda() && output.is_cuda(),
              "fast D22 inputs must be CUDA tensors");
  const auto device = query.device();
  TORCH_CHECK(key.device() == device && value.device() == device &&
                  row_max.device() == device && normalizer.device() == device &&
                  baseline_output.device() == device && positions.device() == device &&
                  probes.device() == device && valid.device() == device &&
                  probe_valid.device() == device && output.device() == device,
              "fast D22 inputs must be on one CUDA device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "fast D22 Q/K/V must have the same dtype");
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16 ||
                  query.scalar_type() == at::kFloat,
              "fast D22 supports float16, bfloat16, and float32 Q/K/V");
  TORCH_CHECK(row_max.scalar_type() == at::kFloat &&
                  normalizer.scalar_type() == at::kFloat &&
                  baseline_output.scalar_type() == at::kFloat &&
                  output.scalar_type() == at::kFloat,
              "fast D22 accumulators and output must be float32");
  TORCH_CHECK(positions.scalar_type() == at::kLong &&
                  probes.scalar_type() == at::kLong,
              "fast D22 positions and probes must be int64");
  TORCH_CHECK(valid.scalar_type() == at::kBool &&
                  probe_valid.scalar_type() == at::kBool,
              "fast D22 validity masks must be bool");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "fast D22 Q/K/V must have rank three");
  TORCH_CHECK(query.size(2) == key.size(2) && query.size(2) == value.size(2),
              "fast D22 head dimensions must match");
  TORCH_CHECK(key.size(0) == value.size(0),
              "fast D22 K/V head counts must match");
  TORCH_CHECK(query.size(0) > 0 && key.size(0) > 0 && query.size(0) % key.size(0) == 0,
              "fast D22 requires divisible GQA head counts");
  TORCH_CHECK(row_max.dim() == 2 && row_max.size(0) == query.size(0) &&
                  row_max.size(1) == query.size(1) &&
                  normalizer.sizes() == row_max.sizes() &&
                  baseline_output.sizes() == query.sizes(),
              "fast D22 baseline shapes do not match Q/K/V");
  TORCH_CHECK(positions.dim() == 1 && probes.dim() == 1 &&
                  valid.dim() == 1 && output.dim() == 1 &&
                  positions.numel() == key.size(1) &&
                  valid.numel() == key.size(1) &&
                  probes.numel() == query.size(1) &&
                  output.numel() == key.size(1) &&
                  probe_valid.sizes() == row_max.sizes(),
              "fast D22 position and validity shapes do not match");
  TORCH_CHECK(query.numel() == 0 || query.stride(2) == 1,
              "fast D22 query must be contiguous in head dimension");
  TORCH_CHECK(key.numel() == 0 || key.stride(2) == 1,
              "fast D22 key must be contiguous in head dimension");
  TORCH_CHECK(value.numel() == 0 || value.stride(2) == 1,
              "fast D22 value must be contiguous in head dimension");
  TORCH_CHECK(row_max.is_contiguous() && normalizer.is_contiguous() &&
                  baseline_output.is_contiguous() && positions.is_contiguous() &&
                  probes.is_contiguous() && valid.is_contiguous() &&
                  probe_valid.is_contiguous() && output.is_contiguous(),
              "fast D22 metadata and output tensors must be contiguous");
  c10::cuda::CUDAGuard device_guard(device);
  makv_fast_d22_chunk_out_cuda(
      query, key, value, row_max, normalizer, baseline_output, positions,
      probes, valid, probe_valid, output, scale_dtype);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(lmcache_makv, m) {
  m.def(
      "fast_d22_chunk_out(Tensor query, Tensor key, Tensor value, "
      "Tensor row_max, Tensor normalizer, Tensor baseline_output, "
      "Tensor positions, Tensor probes, Tensor valid, Tensor probe_valid, "
      "Tensor(a!) output, int scale_dtype) -> ()");
}

TORCH_LIBRARY_IMPL(lmcache_makv, CUDA, m) {
  m.impl("fast_d22_chunk_out", &fast_d22_chunk_out);
}
