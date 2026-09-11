// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Smallest positive normal FP32 value used only to reject a numerically
// singular denominator.  Keeping it local avoids depending on a CUDA macro
// that differs across toolkit releases.
constexpr float kMinNormalFloat = 1.1754943508222875e-38F;

template <typename scalar_t>
__device__ __forceinline__ float load_value(const scalar_t* data,
                                            int64_t index) {
  return static_cast<float>(data[index]);
}

template <typename scalar_t>
__global__ void exact_scalar_d22_kernel(
    const float* __restrict__ q_logits,
    const float* __restrict__ delta_logits,
    const float* __restrict__ output_vhat,
    const float* __restrict__ output_delta_v,
    const float* __restrict__ baseline_output,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ reconstructed_value,
    const float* __restrict__ row_max,
    const float* __restrict__ normalizer,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ probes,
    const bool* __restrict__ valid,
    const bool* __restrict__ probe_valid,
    float* __restrict__ output,
    int64_t kv_heads,
    int64_t groups,
    int64_t probe_count,
    int64_t candidates,
    int64_t head_dim,
    int64_t baseline_stride_head,
    int64_t baseline_stride_probe,
    int64_t baseline_stride_dim,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t reconstructed_stride_head,
    int64_t reconstructed_stride_token,
    int64_t reconstructed_stride_dim) {
  const int64_t candidate = static_cast<int64_t>(blockIdx.x);
  if (candidate >= candidates) {
    return;
  }
  float local = 0.0f;
  const int64_t query_heads = kv_heads * groups;
  const bool candidate_valid = valid[candidate];
  const int64_t position = positions[candidate];
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
    const int64_t group = query_head % groups;
    const int64_t scalar_index =
        ((kv_head * groups + group) * probe_count + probe_index) * candidates +
        candidate;
    const float max_value = row_max[query_head * probe_count + probe_index];
    const float norm = normalizer[query_head * probe_count + probe_index];
    const float q_logit = q_logits[scalar_index];
    const float delta_z = delta_logits[scalar_index];
    const float exp_argument = q_logit - max_value;
    const float attention = expf(exp_argument) / norm;
    if (!isfinite(attention) || !(norm > 0.0f)) {
      continue;
    }
    if (attention <= kMinNormalFloat) {
      continue;
    }

    const int64_t output_base = query_head * baseline_stride_head +
                                probe_index * baseline_stride_probe;
    const int64_t value_base = kv_head * value_stride_head +
                               candidate * value_stride_token;
    const int64_t reconstructed_base = kv_head * reconstructed_stride_head +
                                       candidate * reconstructed_stride_token;
    const int64_t gemm_base = scalar_index;
    float delta_value_norm = 0.0f;
    float reconstructed_minus_output_norm = 0.0f;
    float cross = 0.0f;
    float output_norm = 0.0f;
    float delta_value_dot_reconstructed = 0.0f;
    float reconstructed_norm = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      const float original = load_value(value, value_base + d * value_stride_dim);
      const float reconstructed = load_value(
          reconstructed_value,
          reconstructed_base + d * reconstructed_stride_dim);
      const float base = baseline_output[output_base + d * baseline_stride_dim];
      const float delta_value = reconstructed - original;
      delta_value_norm += delta_value * delta_value;
      reconstructed_norm += reconstructed * reconstructed;
      output_norm += base * base;
      delta_value_dot_reconstructed += delta_value * reconstructed;
    }
    const float output_vhat_value = output_vhat[gemm_base];
    const float output_delta_v_value = output_delta_v[gemm_base];
    reconstructed_minus_output_norm = reconstructed_norm + output_norm -
                                      2.0f * output_vhat_value;
    cross = delta_value_dot_reconstructed - output_delta_v_value;

    if (!isfinite(delta_z)) {
      continue;
    }
    const float u = expm1f(delta_z);
    const float direct_denominator = 1.0f + attention * u;
    const float direct_coefficient = attention / direct_denominator;
    const float direct_bracket = delta_value_norm +
                                 2.0f * u * cross +
                                 u * u * reconstructed_minus_output_norm;
    const float direct_damage = direct_coefficient * direct_coefficient *
                                direct_bracket;
    const bool direct_ok = isfinite(u) && isfinite(direct_denominator) &&
                           direct_denominator > kMinNormalFloat &&
                           isfinite(direct_damage);

    const float inverse_r =
        expf(delta_z > 0.0f ? -delta_z : 0.0f);
    const float stable_denominator =
        (1.0f - attention) * inverse_r + attention;
    const float coefficient_delta_value =
        attention * inverse_r / stable_denominator;
    const float coefficient_value_shift =
        attention * (1.0f - inverse_r) / stable_denominator;
    const float stable_damage =
        coefficient_delta_value * coefficient_delta_value * delta_value_norm +
        2.0f * coefficient_delta_value * coefficient_value_shift * cross +
        coefficient_value_shift * coefficient_value_shift *
            reconstructed_minus_output_norm;
    const bool singular = direct_denominator <= kMinNormalFloat;
    float damage = direct_ok ? direct_damage : stable_damage;
    if (singular) {
      damage = reconstructed_minus_output_norm;
    }
    if (isfinite(damage)) {
      local += fmaxf(0.0f, damage);
    }
  }

  extern __shared__ float reduction[];
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
void launch_exact_scalar_d22(
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
  const int64_t candidates = q_logits.size(3);
  if (candidates == 0 || q_logits.size(2) == 0) {
    return;
  }
  TORCH_CHECK(candidates <= std::numeric_limits<int>::max(),
              "exact scalar D22 chunk is too large for the CUDA launch grid");
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream(q_logits.device().index());
  exact_scalar_d22_kernel<scalar_t><<<
      static_cast<int>(candidates), threads, threads * sizeof(float),
      stream.stream()>>>(
      q_logits.data_ptr<float>(), delta_logits.data_ptr<float>(),
      output_vhat.data_ptr<float>(), output_delta_v.data_ptr<float>(),
      baseline_output.data_ptr<float>(), value.data_ptr<scalar_t>(),
      reconstructed_value.data_ptr<scalar_t>(), row_max.data_ptr<float>(),
      normalizer.data_ptr<float>(), positions.data_ptr<int64_t>(),
      probes.data_ptr<int64_t>(), valid.data_ptr<bool>(),
      probe_valid.data_ptr<bool>(), output.data_ptr<float>(), q_logits.size(0),
      q_logits.size(1), q_logits.size(2), q_logits.size(3),
      baseline_output.size(2), baseline_output.stride(0),
      baseline_output.stride(1), baseline_output.stride(2), value.stride(0),
      value.stride(1), value.stride(2), reconstructed_value.stride(0),
      reconstructed_value.stride(1), reconstructed_value.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

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
    at::Tensor& output) {
  if (value.scalar_type() == at::kHalf) {
    launch_exact_scalar_d22<at::Half>(
        q_logits, delta_logits, output_vhat, output_delta_v, baseline_output,
        value, reconstructed_value, row_max, normalizer, positions, probes,
        valid, probe_valid, output);
  } else if (value.scalar_type() == at::kBFloat16) {
    launch_exact_scalar_d22<at::BFloat16>(
        q_logits, delta_logits, output_vhat, output_delta_v, baseline_output,
        value, reconstructed_value, row_max, normalizer, positions, probes,
        valid, probe_valid, output);
  } else {
    launch_exact_scalar_d22<float>(
        q_logits, delta_logits, output_vhat, output_delta_v, baseline_output,
        value, reconstructed_value, row_max, normalizer, positions, probes,
        valid, probe_valid, output);
  }
}
