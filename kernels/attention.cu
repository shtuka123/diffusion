#include "cuda_utils.h"
#include "tensor.h"
#include "matmul.h"
#include "softmax.h"

#include <cstdio>
#include <cmath>

// Elementwise scale: y = x * c (in-place ok).
__global__ void scale_kernel(const float* x, float* y, float c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * c;
}

// Scaled dot-product attention (single head).
// Q, K: (B, T, d_k). V: (B, T, d_v). Y: (B, T, d_v).
// Causal mask applied.
//
// Intermediates (caller-provided, for reuse across calls):
//   S_buf: (B, T, T)  scores buffer
//   P_buf: (B, T, T)  softmax probabilities buffer
//
// Caller retains P_buf for the backward pass.
void attention_forward(
    Tensor& Y,
    Tensor& P_out,                 // saved for backward
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& S_buf)                 // scratch
{
    int B    = Q.size(0);
    int T    = Q.size(1);
    int d_k  = Q.size(2);
    int d_v  = V.size(2);

    // Shape checks
    if (K.size(0) != B || K.size(1) != T || K.size(2) != d_k ||
        V.size(0) != B || V.size(1) != T || V.size(2) != d_v ||
        Y.size(0) != B || Y.size(1) != T || Y.size(2) != d_v ||
        S_buf.size(0) != B || S_buf.size(1) != T || S_buf.size(2) != T ||
        P_out.size(0) != B || P_out.size(1) != T || P_out.size(2) != T)
    {
        std::fprintf(stderr, "attention_forward: shape mismatch\n");
        std::abort();
    }

    float scale = 1.f / std::sqrt((float)d_k);

    // Per-batch scratch views into Q, K, V, S, P, Y.
    // We launch the matmul once per batch element; batched matmul without
    // an explicit batched kernel is done via a host loop. Wasteful for small
    // per-batch matmuls but simple and correct.
for (int b = 0; b < B; ++b) {
    const float* Q_b = Q.data + b * T * d_k;
    const float* K_b = K.data + b * T * d_k;
    float* S_b = S_buf.data + b * T * T;

    // S_b = Q_b @ K_b^T
    matmul_nt_raw(S_b, Q_b, K_b, T, d_k, T);

    // S_b *= 1 / sqrt(d_k)
    {
        int n = T * T;
        int blk = 256, grd = (n + blk - 1) / blk;
        scale_kernel<<<grd, blk>>>(S_b, S_b, scale, n);
        CUDA_CHECK_KERNEL();
    }
}

    // Causal softmax is already a batched kernel — run once over the whole buffer.
    // S_buf (B, T, T) -> P_out (B, T, T)
    causal_softmax(P_out, S_buf);

    // Y_b = P_b @ V_b, per batch
    // Y_b = P_b @ V_b, per batch
for (int b = 0; b < B; ++b) {
    const float* P_b = P_out.data + b * T * T;
    const float* V_b = V.data + b * T * d_v;
    float* Y_b = Y.data + b * T * d_v;

    // Standard matmul: (T, T) @ (T, d_v) -> (T, d_v)
    matmul_raw(Y_b, P_b, V_b, T, T, d_v);
}
}

// Scaled dot-product attention backward (single-head).
//
// Inputs:
//   dY: (B, T, d_v)  upstream gradient
//   Q, K, V: cached from forward
//   P: cached softmax probabilities from forward
//
// Outputs:
//   dQ: (B, T, d_k)
//   dK: (B, T, d_k)
//   dV: (B, T, d_v)
//
// Scratch (caller-owned, for reuse):
//   dP_buf: (B, T, T)
//   dS_buf: (B, T, T)
void attention_backward(
    Tensor& dQ, Tensor& dK, Tensor& dV,
    const Tensor& dY,
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& P,
    Tensor& dP_buf, Tensor& dS_buf)
{
    int B   = Q.size(0);
    int T   = Q.size(1);
    int d_k = Q.size(2);
    int d_v = V.size(2);
    float scale = 1.f / std::sqrt((float)d_k);

    // Shape checks
    if (dY.size(0) != B || dY.size(1) != T || dY.size(2) != d_v ||
        K.size(1)  != T || K.size(2)  != d_k ||
        V.size(1)  != T ||
        P.size(0)  != B || P.size(1)  != T || P.size(2) != T ||
        dQ.size(0) != B || dQ.size(1) != T || dQ.size(2) != d_k ||
        dK.size(0) != B || dK.size(1) != T || dK.size(2) != d_k ||
        dV.size(0) != B || dV.size(1) != T || dV.size(2) != d_v ||
        dP_buf.size(0) != B || dP_buf.size(1) != T || dP_buf.size(2) != T ||
        dS_buf.size(0) != B || dS_buf.size(1) != T || dS_buf.size(2) != T)
    {
        std::fprintf(stderr, "attention_backward: shape mismatch\n");
        std::abort();
    }

    // Per-batch: loop-launch each matmul, like in attention_forward.
    // Step 1: dV = P^T @ dY
    // Step 2: dP = dY @ V^T
    for (int b = 0; b < B; ++b) {
        const float* P_b  = P.data  + b * T * T;
        const float* dY_b = dY.data + b * T * d_v;
        const float* V_b  = V.data  + b * T * d_v;
        float* dV_b = dV.data      + b * T * d_v;
        float* dP_b = dP_buf.data  + b * T * T;

        // dV = P^T @ dY, shapes (T, T) @ (T, d_v) -> (T, d_v)
        // This is "first operand transposed" — matmul_tn_raw.
        // matmul_tn_raw(C, A, B, M, K, N): C = A^T @ B, with A:(K,M), B:(K,N), C:(M,N)
        // Here: A = P (shape T x T so K=T, M=T), B = dY (shape T x d_v so K=T, N=d_v)
        //       Output dV: (T, d_v)
        matmul_tn_raw(dV_b, P_b, dY_b, T, T, d_v);

        // dP = dY @ V^T, shapes (T, d_v) @ (d_v, T) -> (T, T)
        // "second operand transposed" — matmul_nt_raw.
        // matmul_nt_raw(C, A, B, M, K, N): C = A @ B^T, with A:(M,K), B:(N,K), C:(M,N)
        // Here: A = dY (shape T x d_v so M=T, K=d_v), B = V (shape T x d_v so N=T, K=d_v)
        //       Output dP: (T, T)
        matmul_nt_raw(dP_b, dY_b, V_b, T, d_v, T);
    }

    // Step 3: dS = softmax_backward(P, dP), row-wise, over all (B, T) rows.
    //   This is a batched kernel — runs over the whole P/dP buffers at once.
    softmax_backward_rowwise(dS_buf, P, dP_buf);

    // Step 4: dS *= 1 / sqrt(d_k)
    {
        int n = B * T * T;
        int blk = 256, grd = (n + blk - 1) / blk;
        scale_kernel<<<grd, blk>>>(dS_buf.data, dS_buf.data, scale, n);
        CUDA_CHECK_KERNEL();
    }

    // Step 5: dQ = dS @ K,      shapes (T, T) @ (T, d_k) -> (T, d_k)
    // Step 6: dK = dS^T @ Q,    shapes (T, T) @ (T, d_k) -> (T, d_k) with dS transposed
    for (int b = 0; b < B; ++b) {
        const float* dS_b = dS_buf.data + b * T * T;
        const float* K_b  = K.data      + b * T * d_k;
        const float* Q_b  = Q.data      + b * T * d_k;
        float* dQ_b = dQ.data + b * T * d_k;
        float* dK_b = dK.data + b * T * d_k;

        // dQ = dS @ K
        matmul_raw(dQ_b, dS_b, K_b, T, T, d_k);

        // dK = dS^T @ Q — "first operand transposed"
        matmul_tn_raw(dK_b, dS_b, Q_b, T, T, d_k);
    }
}