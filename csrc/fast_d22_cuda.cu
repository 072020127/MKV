// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <limits>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t* data,
                                             int64_t index) {
  return static_cast<float>(data[index]);
}

template <typename scalar_t>
__device__ __forceinline__ float input_scale(float value) {
  // The reference quantizer forms temporary_scale in the input dtype, then
  // casts that scale to the configured storage dtype for reconstruction.
  return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float stored_scale(float value, bool fp32) {
  if (fp32) {
    return value;
  }
  return static_cast<float>(static_cast<at::Half>(value));
}

template <typename scalar_t>
__global__ void fast_d22_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const float* __restrict__ row_max,
    const float* __restrict__ normalizer,
    const float* __restrict__ baseline_output,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ probes,
    const bool* __restrict__ valid,
    const bool* __restrict__ probe_valid,
    float* __restrict__ output,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t probe_count,
    int64_t chunk_tokens,
    int64_t head_dim,
    int64_t query_stride_head,
    int64_t query_stride_probe,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t value_stride_head,
    int64_t value_stride_token,
    bool scale_fp32) {
  const int64_t candidate = static_cast<int64_t>(blockIdx.x);
  if (candidate >= chunk_tokens) {
    return;
  }

  extern __shared__ float reduction[];
  float* key_reduction = reduction;
  float* value_reduction = reduction + kv_heads * blockDim.x;
  float* key_scales = value_reduction + kv_heads * blockDim.x;
  float* value_scales = key_scales + kv_heads;
  const int64_t key_base = candidate * key_stride_token;
  for (int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    float key_max = 0.0f;
    float value_max = 0.0f;
    const int64_t key_head_base = kv_head * key_stride_head + key_base;
    const int64_t value_head_base =
        kv_head * value_stride_head + candidate * value_stride_token;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      key_max = fmaxf(key_max, fabsf(load_scalar(key, key_head_base + d)));
      value_max =
          fmaxf(value_max, fabsf(load_scalar(value, value_head_base + d)));
    }
    key_reduction[kv_head * blockDim.x + threadIdx.x] = key_max;
    value_reduction[kv_head * blockDim.x + threadIdx.x] = value_max;
  }
  __syncthreads();
  for (unsigned int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      for (int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const int64_t offset = kv_head * blockDim.x + threadIdx.x;
        key_reduction[offset] =
            fmaxf(key_reduction[offset], key_reduction[offset + width]);
        value_reduction[offset] = fmaxf(
            value_reduction[offset], value_reduction[offset + width]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < kv_heads) {
    key_scales[threadIdx.x] = stored_scale(
        key_reduction[threadIdx.x * blockDim.x] == 0.0f
            ? 1.0f
            : key_reduction[threadIdx.x * blockDim.x],
        scale_fp32);
    value_scales[threadIdx.x] = stored_scale(
        value_reduction[threadIdx.x * blockDim.x] == 0.0f
            ? 1.0f
            : value_reduction[threadIdx.x * blockDim.x],
        scale_fp32);
  }
  __syncthreads();

  const int64_t position = positions[candidate];
  const bool candidate_valid = valid[candidate];
  const int64_t groups = query_heads / kv_heads;
  float local = 0.0f;
  const int64_t work = query_heads * probe_count;
  for (int64_t work_index = threadIdx.x; work_index < work;
       work_index += blockDim.x) {
    const int64_t query_head = work_index / probe_count;
    const int64_t probe_index = work_index % probe_count;
    if (!candidate_valid || position >= probes[probe_index] ||
        !probe_valid[query_head * probe_count + probe_index]) {
      continue;
    }
    const int64_t kv_head = query_head / groups;
    const float key_scale = key_scales[kv_head];
    const float value_scale = value_scales[kv_head];
    const float key_quant_scale = input_scale<scalar_t>(
        key_reduction[kv_head * blockDim.x] == 0.0f
            ? 1.0f
            : key_reduction[kv_head * blockDim.x]);
    const float value_quant_scale = input_scale<scalar_t>(
        value_reduction[kv_head * blockDim.x] == 0.0f
            ? 1.0f
            : value_reduction[kv_head * blockDim.x]);
    const float max_value = row_max[query_head * probe_count + probe_index];
    const float norm = normalizer[query_head * probe_count + probe_index];
    if (!(norm > 0.0f) || !isfinite(max_value) || !isfinite(norm)) {
      continue;
    }

    float original_logit = 0.0f;
    float delta_logit = 0.0f;
    const int64_t query_base =
        query_head * query_stride_head + probe_index * query_stride_probe;
    const int64_t key_head_base = kv_head * key_stride_head + key_base;
    const int64_t value_head_base = kv_head * value_stride_head +
                                    candidate * value_stride_token;
    const int64_t output_base =
        (query_head * probe_count + probe_index) * head_dim;
    for (int64_t d = 0; d < head_dim; ++d) {
      const float q = load_scalar(query, query_base + d);
      const float key_value = load_scalar(key, key_head_base + d);
      const float value_value = load_scalar(value, value_head_base + d);
      const float key_q = rintf(key_value / key_quant_scale);
      const float value_q = rintf(value_value / value_quant_scale);
      const float key_hat = fminf(1.0f, fmaxf(-1.0f, key_q)) * key_scale;
      const float value_hat =
          fminf(1.0f, fmaxf(-1.0f, value_q)) * value_scale;
      original_logit += q * key_value;
      delta_logit += q * (key_hat - key_value);
    }
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const float logit = original_logit * attention_scale;
    const float delta = delta_logit * attention_scale;
    float attention = expf(logit - max_value) / norm;
    const float bounded_delta = fminf(delta, 88.72283172607422f);
    const float ratio = expf(bounded_delta);
    const bool use_high_branch = ratio > 1.0f;
    const float denominator = (1.0f - attention) + attention * ratio;
    const bool singular = denominator == 0.0f;
    const float safe_denominator = singular ? 1.0f : denominator;
    const float inverse_ratio = use_high_branch ? 1.0f / ratio : 1.0f;
    const float high_denominator =
        (1.0f - attention) * inverse_ratio + attention;
    for (int64_t d = 0; d < head_dim; ++d) {
      const float value_value = load_scalar(value, value_head_base + d);
      const float value_q = rintf(value_value / value_quant_scale);
      const float value_hat =
          fminf(1.0f, fmaxf(-1.0f, value_q)) * value_scale;
      const float base_output = baseline_output[output_base + d];
      float updated;
      if (use_high_branch) {
        const float numerator =
            (base_output - attention * value_value) * inverse_ratio +
            attention * value_hat;
        const float high_safe_denominator =
            singular ? 1.0f : high_denominator;
        updated = numerator / high_safe_denominator;
      } else {
        const float numerator =
            base_output - attention * value_value +
            attention * ratio * value_hat;
        updated = numerator / safe_denominator;
      }
      if (singular) {
        updated = value_hat;
      }
      const float difference = updated - base_output;
      local += difference * difference;
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      reduction[threadIdx.x] += reduction[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[candidate] = reduction[0];
  }
}

template <typename scalar_t>
void launch_fast_d22(
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
  constexpr int threads = 256;
  const int64_t candidates = key.size(1);
  const int64_t kv_heads = key.size(0);
  if (candidates == 0 || query.size(1) == 0) {
    return;
  }
  TORCH_CHECK(candidates <= std::numeric_limits<int>::max(),
              "fast D22 chunk is too large for the CUDA launch grid");
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  const size_t shared_bytes =
      (2 * kv_heads * threads + 2 * kv_heads) * sizeof(float);
  fast_d22_kernel<scalar_t><<<static_cast<int>(candidates), threads,
                              shared_bytes, stream.stream()>>>(
      query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
      value.data_ptr<scalar_t>(), row_max.data_ptr<float>(),
      normalizer.data_ptr<float>(), baseline_output.data_ptr<float>(),
      positions.data_ptr<int64_t>(), probes.data_ptr<int64_t>(),
      valid.data_ptr<bool>(), probe_valid.data_ptr<bool>(),
      output.data_ptr<float>(), query.size(0), key.size(0), query.size(1),
      key.size(1), query.size(2), query.stride(0), query.stride(1),
      key.stride(0), key.stride(1), value.stride(0), value.stride(1),
      scale_dtype != 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

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
    int64_t scale_dtype) {
  if (query.scalar_type() == at::kHalf) {
    launch_fast_d22<at::Half>(
        query, key, value, row_max, normalizer, baseline_output, positions,
        probes, valid, probe_valid, output, scale_dtype);
  } else if (query.scalar_type() == at::kBFloat16) {
    launch_fast_d22<at::BFloat16>(
        query, key, value, row_max, normalizer, baseline_output, positions,
        probes, valid, probe_valid, output, scale_dtype);
  } else {
    launch_fast_d22<float>(
        query, key, value, row_max, normalizer, baseline_output, positions,
        probes, valid, probe_valid, output, scale_dtype);
  }
}
