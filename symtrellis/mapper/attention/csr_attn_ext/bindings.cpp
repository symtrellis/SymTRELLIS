#include <torch/extension.h>

#include <vector>

// Thin pybind layer for the CSR attention CUDA implementation.
//
// Python owns user-facing validation, q/k/v projection, RoPE, q/k RMSNorm, and
// output projection. These bindings only expose the CUDA launcher functions.

// Forward directed CSR attention over already-projected q/k/v:
//   q:      [Nq, H, D]
//   k, v:   [Nk, H, D]
//   rowptr: [Nq + 1]
//   col:    [E]
// Returns {out [Nq, H, D], lse [Nq, H]}.
std::vector<torch::Tensor> sparse_csr_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rowptr,
    torch::Tensor col);

// Backward directed CSR attention. Uses saved forward out/lse and recomputes
// edge scores/probabilities; returns {dq, dk, dv}. dK/dV are accumulated with
// CUDA atomics inside the kernel.
std::vector<torch::Tensor> sparse_csr_attn_backward(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor out,
    torch::Tensor lse);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "sparse_csr_attn_forward",
        &sparse_csr_attn_forward,
        "Directed CSR attention forward: (q, k, v, rowptr, col) -> (out, lse)");
    m.def(
        "sparse_csr_attn_backward",
        &sparse_csr_attn_backward,
        "Directed CSR attention backward: "
        "(grad_out, q, k, v, rowptr, col, out, lse) -> (dq, dk, dv)");
}
