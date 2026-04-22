#include "cuda_utils.h"
#include "tensor.h"
#include "matmul.h"
#include "linear.h"
#include "attention.h"
#include "transpose.h"

#include <cstdio>

// Multi-head self-attention forward.
//
// Shapes:
//   x:   (B, T, D)    where D = n_heads * d_k
//   y:   (B, T, D)    output
//   W_Q, W_K, W_V: (D, D)   input projection weights
//   b_Q, b_K, b_V: (D,)     input projection biases
//   W_O: (D, D), b_O: (D,)  output projection
//
// Scratch buffers (caller-owned, reused across calls):
//   Q_raw, K_raw, V_raw: (B, T, D)          — post-projection, pre-reshape
//   Q_heads, K_heads, V_heads: (B, H, T, dk) — post-transpose
//   attn_out_heads: (B, H, T, dk)
//   attn_out_flat:  (B, T, D)
//   S_buf: (B*H, T, T)  — attention scores scratch
//
// Saved for backward (caller-owned):
//   P_save: (B*H, T, T) — softmax probs across all heads/batches
struct MHACache {
    // Saved for backward
    Tensor Q_heads, K_heads, V_heads;   // (B, H, T, dk)
    Tensor P;                           // (B*H, T, T)
    Tensor attn_out_flat;               // (B, T, D) — before W_O
    // Scratch (not strictly needed for backward but kept here for reuse)
    Tensor Q_raw, K_raw, V_raw;         // (B, T, D)
    Tensor attn_out_heads;              // (B, H, T, dk)
    Tensor S_buf;                       // (B*H, T, T)
};

void mha_forward(
    Tensor& y,
    MHACache& cache,
    const Tensor& x,
    const Tensor& W_Q, const Tensor& b_Q,
    const Tensor& W_K, const Tensor& b_K,
    const Tensor& W_V, const Tensor& b_V,
    const Tensor& W_O, const Tensor& b_O,
    int n_heads)
{
    int B = x.size(0);
    int T = x.size(1);
    int D = x.size(2);
    int d_k = D / n_heads;
    if (D % n_heads != 0) {
        std::fprintf(stderr, "mha_forward: D (%d) not divisible by n_heads (%d)\n",
                     D, n_heads);
        std::abort();
    }
    int H = n_heads;

    // ----- Input projections: Q/K/V = Linear(x) -----
    // linear_forward expects x as 2D (B', D_in). Here x is (B, T, D) which is
    // effectively (B*T, D) flattened — we can just treat it as 2D.
    // Our linear_forward takes (B, D_in) so we reshape-view via shape mutation.
    // Cheapest: create a tensor-like wrapper via raw pointer call since all
    // we need is matmul + bias_add.
    //
    // Simplest path: temporarily reshape Tensors by mutating shape (numel stays).
    // Because our Tensor class doesn't have a view() method yet, we take the
    // raw route and do the matmul+bias_add via sub-operations.

    // We assume cache.Q_raw is (B, T, D); reshape conceptually to (B*T, D).
    // linear_forward accepts 2D input only, so we fake 2D by wrapping.
    // Easiest hack: use a local helper that just calls matmul_raw + elementwise add.
    // But we already have linear_forward taking Tensor. Simpler: reshape x in place.

    // We'll accept a minor inelegance here: temporarily mutate shapes to 2D,
    // call linear_forward, then mutate back.
    //
    // NOTE: This mutates the *cache's* tensor shape metadata. Since Tensor's
    // `shape` is a vector<int>, this is cheap and safe as long as numel is preserved.

    auto linear_3d = [](Tensor& y_3d, const Tensor& x_3d,
                        const Tensor& W, const Tensor& b)
    {
        int B = x_3d.size(0), T = x_3d.size(1);
        int D_in = x_3d.size(2), D_out = W.size(1);
        // y = x @ W    treating x as (B*T, D_in) and y as (B*T, D_out)
        matmul_raw(y_3d.data, x_3d.data, W.data, B * T, D_in, D_out);
        // y += b (broadcast b over rows)
        // We can call our bias_add directly since bias_add's kernel uses i % feat_dim
        // which works for 2D (B*T, D_out) as well.
        // But bias_add takes Tensor&, and we want to avoid reshaping;
        // luckily bias_add's logic is numel-agnostic — it just needs feat_dim.
        // For now, inline via a small helper call:
        extern __global__ void bias_add_kernel(const float*, const float*, float*,
                                               int, int);
        int n = B * T * D_out;
        int blk = 256, grd = (n + blk - 1) / blk;
        bias_add_kernel<<<grd, blk>>>(y_3d.data, b.data, y_3d.data, n, D_out);
        CUDA_CHECK_KERNEL();
    };

    linear_3d(cache.Q_raw, x, W_Q, b_Q);
    linear_3d(cache.K_raw, x, W_K, b_K);
    linear_3d(cache.V_raw, x, W_V, b_V);

    // ----- Reshape + transpose to per-head layout -----
    // Q_raw: (B, T, D) viewed as (B, T, H, dk), transpose to (B, H, T, dk).
    // View is free (just mutate shape metadata).
    auto reshape_for_heads = [&](Tensor& t) {
        t.shape = {B, T, H, d_k};
    };
    auto restore_3d = [&](Tensor& t) {
        t.shape = {B, T, D};
    };

    reshape_for_heads(cache.Q_raw);
    transpose_12(cache.Q_heads, cache.Q_raw);
    restore_3d(cache.Q_raw);

    reshape_for_heads(cache.K_raw);
    transpose_12(cache.K_heads, cache.K_raw);
    restore_3d(cache.K_raw);

    reshape_for_heads(cache.V_raw);
    transpose_12(cache.V_heads, cache.V_raw);
    restore_3d(cache.V_raw);

    // ----- Run attention with (B*H) as effective batch dim -----
    // cache.Q_heads is (B, H, T, dk). We view it as (B*H, T, dk) for the
    // attention call.
    auto flatten_bh = [&](Tensor& t) {
        t.shape = {B * H, T, d_k};
    };
    auto unflatten_bh = [&](Tensor& t) {
        t.shape = {B, H, T, d_k};
    };

    flatten_bh(cache.Q_heads);
    flatten_bh(cache.K_heads);
    flatten_bh(cache.V_heads);

    // attn_out_heads will receive (B*H, T, dk); also flatten it.
    cache.attn_out_heads.shape = {B * H, T, d_k};

    attention_forward(
        cache.attn_out_heads,
        cache.P,
        cache.Q_heads, cache.K_heads, cache.V_heads,
        cache.S_buf);

    // Restore 4D shape
    unflatten_bh(cache.Q_heads);
    unflatten_bh(cache.K_heads);
    unflatten_bh(cache.V_heads);
    cache.attn_out_heads.shape = {B, H, T, d_k};

    // ----- Transpose back and reshape to (B, T, D) -----
    // attn_out_heads: (B, H, T, dk) -> (B, T, H, dk) via transpose_12
    // then view as (B, T, D)
    // transpose_12 swaps dims 1 and 2 — exactly what we need.
    {
        // We need a (B, T, H, dk) output. attn_out_flat has shape (B, T, D).
        // We view attn_out_flat as (B, T, H, dk) for the transpose output.
        cache.attn_out_flat.shape = {B, T, H, d_k};
        transpose_12(cache.attn_out_flat, cache.attn_out_heads);
        cache.attn_out_flat.shape = {B, T, D};
    }

    // ----- Output projection: y = attn_out_flat @ W_O + b_O -----
    linear_3d(y, cache.attn_out_flat, W_O, b_O);
}   