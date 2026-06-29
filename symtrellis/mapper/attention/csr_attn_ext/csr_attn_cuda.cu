#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cmath>
#include <cstdint>
#include <vector>

// Directed CSR scaled dot-product attention over already-projected q/k/v.
//
// For query row i and head h, this computes attention over the key/value rows
// listed by col[rowptr[i] : rowptr[i + 1]]. The CUDA backend does not build the
// graph, sort rows, apply windows, or apply positional encodings. It also does
// not materialize score [E, H], probability [E, H], or message [E, H, D]
// tensors; backward recomputes scores and probabilities from q/k/lse.
//
// One warp computes one (query row, head). The kernel accepts arbitrary CSR row
// lengths, but is intended for short rows such as average degree around 10 and
// max degree around 88. No high-degree or degree-specific path is implemented.

namespace
{
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  __device__ __forceinline__ float warp_sum(float x)
  {
    // All-reduce a lane-local partial dot product and broadcast the scalar to
    // every lane. The kernels use this for q.k, grad_out.out, and grad_out.v.
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
      x += __shfl_down_sync(kFullWarpMask, x, offset);
    }
    return __shfl_sync(kFullWarpMask, x, 0);
  }

  template <typename scalar_t>
  __device__ __forceinline__ void atomic_add(scalar_t *ptr, float value)
  {
    // dK/dV are many-to-one accumulations across query rows. Use ATen CUDA
    // atomic helpers for at::Half and at::BFloat16 instead of hand-written CAS.
    gpuAtomicAddNoReturn(ptr, static_cast<scalar_t>(value));
  }

  template <>
  __device__ __forceinline__ void atomic_add<float>(float *ptr, float value)
  {
    // fp32 has native CUDA atomic add support.
    atomicAdd(ptr, value);
  }

  // Forward CSR attention.
  //
  // Inputs:
  //   q:      [Nq, H, HEAD_DIM], projected query features.
  //   k, v:   [Nk, H, HEAD_DIM], projected key/value features.
  //   rowptr: [Nq + 1], int32 CSR row pointer.
  //   col:    [E], int32 key/value row indices.
  // Outputs:
  //   out: [Nq, H, HEAD_DIM], attended value features.
  //   lse: [Nq, H], float32 log-sum-exp for backward.
  //
  // Math for one (i, h):
  //   score_e = dot(q[i,h], k[col[e],h]) / sqrt(HEAD_DIM)
  //   p_e = softmax_e(score_e)
  //   out[i,h] = sum_e p_e * v[col[e],h]
  template <typename scalar_t, int HEAD_DIM, int WARPS_PER_BLOCK>
  __global__ void csr_attn_forward_kernel(
      const scalar_t *__restrict__ q,
      const scalar_t *__restrict__ k,
      const scalar_t *__restrict__ v,
      const int32_t *__restrict__ rowptr,
      const int32_t *__restrict__ col,
      scalar_t *__restrict__ out,
      float *__restrict__ lse,
      int64_t Nq,
      int64_t H,
      float scale)
  {
    constexpr int kSlots = (HEAD_DIM + kWarpSize - 1) / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t warp_id =
        (static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK) + warp_in_block;
    const int64_t total_warps = Nq * H;
    if (warp_id >= total_warps)
    {
      return;
    }

    // Map this warp to one query row and one attention head.
    const int64_t i = warp_id / H;
    const int64_t h = warp_id - i * H;
    const int32_t start = rowptr[i];
    const int32_t end = rowptr[i + 1];
    const int64_t q_base = (i * H + h) * HEAD_DIM;

    if (start == end)
    {
      // Empty rows have zero attention output and zero lse by definition.
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          out[q_base + d] = static_cast<scalar_t>(0.0f);
        }
      }
      if (lane == 0)
      {
        lse[i * H + h] = 0.0f;
      }
      return;
    }

    // Query values are reused for every edge in this CSR row. Keep each lane's
    // owned channels in registers rather than rereading q inside the edge loop.
    float q_reg[kSlots];
    for (int slot = 0; slot < kSlots; ++slot)
    {
      const int d = lane + slot * kWarpSize;
      q_reg[slot] = d < HEAD_DIM ? static_cast<float>(q[q_base + d]) : 0.0f;
    }

    // First pass: compute the stable softmax max m = max_e score_e.
    float m = -INFINITY;
    for (int32_t e = start; e < end; ++e)
    {
      const int32_t j = col[e];
      const int64_t kv_base = (static_cast<int64_t>(j) * H + h) * HEAD_DIM;

      float partial = 0.0f;
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          partial += q_reg[slot] * static_cast<float>(k[kv_base + d]);
        }
      }
      const float score = warp_sum(partial) * scale;
      m = fmaxf(m, score);
    }

    // Second pass: compute z = sum_e exp(score_e - m) and the unnormalized
    // value accumulator sum_e exp(score_e - m) * v[col[e], h].
    float z = 0.0f;
    float acc[kSlots];
    for (int slot = 0; slot < kSlots; ++slot)
    {
      acc[slot] = 0.0f;
    }

    for (int32_t e = start; e < end; ++e)
    {
      const int32_t j = col[e];
      const int64_t kv_base = (static_cast<int64_t>(j) * H + h) * HEAD_DIM;

      float partial = 0.0f;
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          partial += q_reg[slot] * static_cast<float>(k[kv_base + d]);
        }
      }
      const float score = warp_sum(partial) * scale;
      const float w = expf(score - m);
      z += w;

      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          acc[slot] += w * static_cast<float>(v[kv_base + d]);
        }
      }
    }

    // Normalize the value accumulator and save lse = m + log(z) for backward.
    const float inv_z = 1.0f / z;
    for (int slot = 0; slot < kSlots; ++slot)
    {
      const int d = lane + slot * kWarpSize;
      if (d < HEAD_DIM)
      {
        out[q_base + d] = static_cast<scalar_t>(acc[slot] * inv_z);
      }
    }
    if (lane == 0)
    {
      lse[i * H + h] = m + logf(z);
    }
  }

  // Backward CSR attention.
  //
  // Inputs:
  //   grad_out: [Nq, H, HEAD_DIM], dL/dout.
  //   q, k, v:  same tensors used by forward.
  //   rowptr, col: CSR edge structure used by forward.
  //   out: [Nq, H, HEAD_DIM], saved forward output.
  //   lse: [Nq, H], saved forward log-sum-exp.
  // Outputs:
  //   dq: [Nq, H, HEAD_DIM]
  //   dk: [Nk, H, HEAD_DIM]
  //   dv: [Nk, H, HEAD_DIM]
  //
  // Math for one edge i -> j in head h:
  //   delta = dot(grad_out[i,h], out[i,h])
  //   p = exp(score(i,j,h) - lse[i,h])
  //   u = dot(grad_out[i,h], v[j,h])
  //   ds = p * (u - delta)
  //   dq[i,h] += ds * k[j,h] / sqrt(HEAD_DIM)
  //   dk[j,h] += ds * q[i,h] / sqrt(HEAD_DIM)
  //   dv[j,h] += p * grad_out[i,h]
  template <typename scalar_t, int HEAD_DIM, int WARPS_PER_BLOCK>
  __global__ void csr_attn_backward_kernel(
      const scalar_t *__restrict__ grad_out,
      const scalar_t *__restrict__ q,
      const scalar_t *__restrict__ k,
      const scalar_t *__restrict__ v,
      const int32_t *__restrict__ rowptr,
      const int32_t *__restrict__ col,
      const scalar_t *__restrict__ out,
      const float *__restrict__ lse,
      scalar_t *__restrict__ dq,
      scalar_t *__restrict__ dk,
      scalar_t *__restrict__ dv,
      int64_t Nq,
      int64_t H,
      float scale)
  {
    constexpr int kSlots = (HEAD_DIM + kWarpSize - 1) / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t warp_id =
        (static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK) + warp_in_block;
    const int64_t total_warps = Nq * H;
    if (warp_id >= total_warps)
    {
      return;
    }

    // Map this warp to one query row and one attention head.
    const int64_t i = warp_id / H;
    const int64_t h = warp_id - i * H;
    const int32_t start = rowptr[i];
    const int32_t end = rowptr[i + 1];
    const int64_t q_base = (i * H + h) * HEAD_DIM;

    if (start == end)
    {
      // Empty rows only contribute zero dQ and no dK/dV updates.
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          dq[q_base + d] = static_cast<scalar_t>(0.0f);
        }
      }
      return;
    }

    // Query-side values are reused across every edge in this row. grad_out and
    // out are also reused for delta and dV, so keep all three in registers.
    float q_reg[kSlots];
    float go_reg[kSlots];
    float out_reg[kSlots];
    for (int slot = 0; slot < kSlots; ++slot)
    {
      const int d = lane + slot * kWarpSize;
      if (d < HEAD_DIM)
      {
        q_reg[slot] = static_cast<float>(q[q_base + d]);
        go_reg[slot] = static_cast<float>(grad_out[q_base + d]);
        out_reg[slot] = static_cast<float>(out[q_base + d]);
      }
      else
      {
        q_reg[slot] = 0.0f;
        go_reg[slot] = 0.0f;
        out_reg[slot] = 0.0f;
      }
    }

    // Softmax backward needs delta = dot(grad_out, out) for this row/head.
    float delta_partial = 0.0f;
    for (int slot = 0; slot < kSlots; ++slot)
    {
      const int d = lane + slot * kWarpSize;
      if (d < HEAD_DIM)
      {
        delta_partial += go_reg[slot] * out_reg[slot];
      }
    }
    const float delta = warp_sum(delta_partial);
    const float lse_val = lse[i * H + h];

    float dq_acc[kSlots];
    for (int slot = 0; slot < kSlots; ++slot)
    {
      dq_acc[slot] = 0.0f;
    }

    for (int32_t e = start; e < end; ++e)
    {
      const int32_t j = col[e];
      const int64_t kv_base = (static_cast<int64_t>(j) * H + h) * HEAD_DIM;

      // Recompute score and p from q/k/lse instead of loading saved scores or
      // probabilities. Also compute u = dot(grad_out, v) for this edge.
      float score_partial = 0.0f;
      float u_partial = 0.0f;
      float k_reg[kSlots];
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          k_reg[slot] = static_cast<float>(k[kv_base + d]);
          score_partial += q_reg[slot] * k_reg[slot];
          u_partial += go_reg[slot] * static_cast<float>(v[kv_base + d]);
        }
        else
        {
          k_reg[slot] = 0.0f;
        }
      }

      const float score = warp_sum(score_partial) * scale;
      const float p = expf(score - lse_val);
      const float u = warp_sum(u_partial);
      const float ds = p * (u - delta);

      // dQ is row-local and can be accumulated in registers. dK/dV receive
      // contributions from many query rows and therefore use atomic add.
      for (int slot = 0; slot < kSlots; ++slot)
      {
        const int d = lane + slot * kWarpSize;
        if (d < HEAD_DIM)
        {
          dq_acc[slot] += ds * k_reg[slot] * scale;
          atomic_add(&dk[kv_base + d], ds * q_reg[slot] * scale);
          atomic_add(&dv[kv_base + d], p * go_reg[slot]);
        }
      }
    }

    // Each (query row, head, channel) is owned by exactly one warp/lane.
    for (int slot = 0; slot < kSlots; ++slot)
    {
      const int d = lane + slot * kWarpSize;
      if (d < HEAD_DIM)
      {
        dq[q_base + d] = static_cast<scalar_t>(dq_acc[slot]);
      }
    }
  }

} // namespace

// Allocate forward outputs through PyTorch's CUDA allocator, dispatch only the
// supported dtype/head_dim specializations, and launch the forward kernel.
//
// Inputs are assumed to have been validated by the Python boundary:
//   q:      [Nq, H, D], CUDA contiguous, fp32/fp16/bf16.
//   k, v:   [Nk, H, D], same dtype as q.
//   rowptr: [Nq + 1], CUDA contiguous int32.
//   col:    [E], CUDA contiguous int32.
// Returns:
//   {out, lse}, where out is [Nq, H, D] in q's dtype and lse is [Nq, H] fp32.
std::vector<torch::Tensor> sparse_csr_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rowptr,
    torch::Tensor col)
{
  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  const int64_t Nq = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  auto out = at::empty_like(q);
  auto lse = at::empty({Nq, H}, q.options().dtype(at::kFloat));

  const int64_t total_warps = Nq * H;
  if (total_warps == 0)
  {
    return {out, lse};
  }

  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// Dispatch is intentionally limited to the public contract: fp32/fp16/bf16 and
// head dimensions 16/32/64/128.
#define LAUNCH_FORWARD(SCALAR_T, HEAD_DIM)                                               \
  csr_attn_forward_kernel<SCALAR_T, HEAD_DIM, kWarpsPerBlock>                            \
      <<<grid, block, 0, stream>>>(                                                      \
          q.data_ptr<SCALAR_T>(), k.data_ptr<SCALAR_T>(), v.data_ptr<SCALAR_T>(),        \
          rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(), out.data_ptr<SCALAR_T>(), \
          lse.data_ptr<float>(), Nq, H, scale)

#define DISPATCH_FORWARD(SCALAR_T)                                                     \
  do                                                                                   \
  {                                                                                    \
    switch (D)                                                                         \
    {                                                                                  \
    case 16:                                                                           \
      LAUNCH_FORWARD(SCALAR_T, 16);                                                    \
      break;                                                                           \
    case 32:                                                                           \
      LAUNCH_FORWARD(SCALAR_T, 32);                                                    \
      break;                                                                           \
    case 64:                                                                           \
      LAUNCH_FORWARD(SCALAR_T, 64);                                                    \
      break;                                                                           \
    case 128:                                                                          \
      LAUNCH_FORWARD(SCALAR_T, 128);                                                   \
      break;                                                                           \
    default:                                                                           \
      TORCH_CHECK(false, "unsupported head_dim passed to CSR attention CUDA forward"); \
    }                                                                                  \
  } while (false)

  switch (q.scalar_type())
  {
  case at::kFloat:
    DISPATCH_FORWARD(float);
    break;
  case at::kHalf:
    DISPATCH_FORWARD(at::Half);
    break;
  case at::kBFloat16:
    DISPATCH_FORWARD(at::BFloat16);
    break;
  default:
    TORCH_CHECK(false, "CSR attention supports only float32, float16, bfloat16");
  }

#undef DISPATCH_FORWARD
#undef LAUNCH_FORWARD

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

// Allocate backward outputs through PyTorch's CUDA allocator, dispatch only the
// supported dtype/head_dim specializations, and launch the backward kernel.
//
// Inputs:
//   grad_out: [Nq, H, D], same dtype as q.
//   q:        [Nq, H, D]
//   k, v:     [Nk, H, D]
//   rowptr:   [Nq + 1], int32.
//   col:      [E], int32.
//   out:      [Nq, H, D], saved forward output.
//   lse:      [Nq, H], saved forward log-sum-exp in fp32.
// Returns:
//   {dq, dk, dv}, with the same shapes and dtypes as q, k, and v.
// Note: dK/dV accumulation uses atomics and is not bitwise deterministic.
std::vector<torch::Tensor> sparse_csr_attn_backward(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor out,
    torch::Tensor lse)
{
  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(q));

  const int64_t Nq = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  auto dq = at::empty_like(q);
  auto dk = at::zeros_like(k);
  auto dv = at::zeros_like(v);

  const int64_t total_warps = Nq * H;
  if (total_warps == 0)
  {
    return {dq, dk, dv};
  }

  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// Dispatch is intentionally limited to the public contract: fp32/fp16/bf16 and
// head dimensions 16/32/64/128.
#define LAUNCH_BACKWARD(SCALAR_T, HEAD_DIM)                                           \
  csr_attn_backward_kernel<SCALAR_T, HEAD_DIM, kWarpsPerBlock>                        \
      <<<grid, block, 0, stream>>>(                                                   \
          grad_out.data_ptr<SCALAR_T>(), q.data_ptr<SCALAR_T>(),                      \
          k.data_ptr<SCALAR_T>(), v.data_ptr<SCALAR_T>(), rowptr.data_ptr<int32_t>(), \
          col.data_ptr<int32_t>(), out.data_ptr<SCALAR_T>(), lse.data_ptr<float>(),   \
          dq.data_ptr<SCALAR_T>(), dk.data_ptr<SCALAR_T>(), dv.data_ptr<SCALAR_T>(),  \
          Nq, H, scale)

#define DISPATCH_BACKWARD(SCALAR_T)                                                     \
  do                                                                                    \
  {                                                                                     \
    switch (D)                                                                          \
    {                                                                                   \
    case 16:                                                                            \
      LAUNCH_BACKWARD(SCALAR_T, 16);                                                    \
      break;                                                                            \
    case 32:                                                                            \
      LAUNCH_BACKWARD(SCALAR_T, 32);                                                    \
      break;                                                                            \
    case 64:                                                                            \
      LAUNCH_BACKWARD(SCALAR_T, 64);                                                    \
      break;                                                                            \
    case 128:                                                                           \
      LAUNCH_BACKWARD(SCALAR_T, 128);                                                   \
      break;                                                                            \
    default:                                                                            \
      TORCH_CHECK(false, "unsupported head_dim passed to CSR attention CUDA backward"); \
    }                                                                                   \
  } while (false)

  switch (q.scalar_type())
  {
  case at::kFloat:
    DISPATCH_BACKWARD(float);
    break;
  case at::kHalf:
    DISPATCH_BACKWARD(at::Half);
    break;
  case at::kBFloat16:
    DISPATCH_BACKWARD(at::BFloat16);
    break;
  default:
    TORCH_CHECK(false, "CSR attention supports only float32, float16, bfloat16");
  }

#undef DISPATCH_BACKWARD
#undef LAUNCH_BACKWARD

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}
