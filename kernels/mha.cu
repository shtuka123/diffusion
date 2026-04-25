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

// MHA backward.
//
// Inputs:
//   dy:    (B, T, D)  upstream gradient
//   x:     (B, T, D)  original input (cached or re-passed)
//   cache: from forward — has Q_heads, K_heads, V_heads, P, attn_out_flat, etc.
//   W_Q, W_K, W_V, W_O: weight matrices (need shapes for backward)
//
// Outputs:
//   dx:    (B, T, D)
//   dW_Q, db_Q: (D, D), (D,)
//   dW_K, db_K: (D, D), (D,)
//   dW_V, db_V: (D, D), (D,)
//   dW_O, db_O: (D, D), (D,)
//
// Scratch (allocated locally):
//   d_attn_out_flat: (B, T, D)
//   d_attn_out_heads: (B, H, T, dk)
//   dQ_heads, dK_heads, dV_heads: (B*H, T, dk) when flat, (B, H, T, dk) when not
//   dQ_raw, dK_raw, dV_raw: (B, T, D)
//   dP_buf, dS_buf: (B*H, T, T) — for attention_backward

void mha_backward(
    Tensor& dx,
    Tensor& dW_Q, Tensor& db_Q,
    Tensor& dW_K, Tensor& db_K,
    Tensor& dW_V, Tensor& db_V,
    Tensor& dW_O, Tensor& db_O,
    const Tensor& dy,
    const Tensor& x,
    MHACache& cache,
    const Tensor& W_Q, const Tensor& W_K,
    const Tensor& W_V, const Tensor& W_O,
    int n_heads)
{
    int B = x.size(0), T = x.size(1), D = x.size(2);
    int H = n_heads;
    int d_k = D / H;

    // Helpers
    auto to2d = [](Tensor& t, int rows, int cols) {
        t.shape = {rows, cols};
    };
    auto to3d = [](Tensor& t, int B, int T, int D) {
        t.shape = {B, T, D};
    };
    auto to4d = [](Tensor& t, int d0, int d1, int d2, int d3) {
        t.shape = {d0, d1, d2, d3};
    };

    // ===== Step 1: backward through output projection =====
    // y = attn_out_flat @ W_O + b_O
    // -> dW_O = attn_out_flat^T @ dy
    //    db_O = sum(dy, axis=batch)
    //    d_attn_out_flat = dy @ W_O^T

    Tensor d_attn_out_flat({B, T, D});
    {
        Tensor& aof = cache.attn_out_flat;
        Tensor& dy_nc = const_cast<Tensor&>(dy);

        to2d(aof, B * T, D);
        to2d(dy_nc, B * T, D);
        to2d(d_attn_out_flat, B * T, D);

        linear_backward(d_attn_out_flat, dW_O, db_O, aof, W_O, dy_nc);

        to3d(aof, B, T, D);
        to3d(dy_nc, B, T, D);
        to3d(d_attn_out_flat, B, T, D);
    }

    // ===== Step 2: backward through reshape + transpose =====
    // attn_out_heads (B, H, T, dk) -> transpose_12 -> (B, T, H, dk) -> view (B, T, D)
    // Backward: d_attn_out_flat (B, T, D) -> view (B, T, H, dk) -> transpose_12 -> (B, H, T, dk)
    Tensor d_attn_out_heads({B, H, T, d_k});
    {
        // View d_attn_out_flat as (B, T, H, dk), transpose to (B, H, T, dk).
        Tensor& src = d_attn_out_flat;
        to4d(src, B, T, H, d_k);
        transpose_12(d_attn_out_heads, src);
        to3d(src, B, T, D);
    }

    // ===== Step 3: backward through attention =====
    // attention_forward took (B*H, T, dk) for Q, K, V. Backward needs the
    // same flattened layout.
    Tensor dQ_heads({B * H, T, d_k});
    Tensor dK_heads({B * H, T, d_k});
    Tensor dV_heads({B * H, T, d_k});
    Tensor dP_buf({B * H, T, T});
    Tensor dS_buf({B * H, T, T});

    // Flatten d_attn_out_heads from (B, H, T, dk) to (B*H, T, dk).
    {
        d_attn_out_heads.shape = {B * H, T, d_k};
    }
    // Flatten Q_heads / K_heads / V_heads (they're stored as (B, H, T, dk) in cache)
    cache.Q_heads.shape = {B * H, T, d_k};
    cache.K_heads.shape = {B * H, T, d_k};
    cache.V_heads.shape = {B * H, T, d_k};

    attention_backward(
        dQ_heads, dK_heads, dV_heads,
        d_attn_out_heads,
        cache.Q_heads, cache.K_heads, cache.V_heads,
        cache.P,
        dP_buf, dS_buf);

    // Restore 4D shapes
    cache.Q_heads.shape = {B, H, T, d_k};
    cache.K_heads.shape = {B, H, T, d_k};
    cache.V_heads.shape = {B, H, T, d_k};
    dQ_heads.shape = {B, H, T, d_k};
    dK_heads.shape = {B, H, T, d_k};
    dV_heads.shape = {B, H, T, d_k};

    // ===== Step 4: backward through transpose + reshape =====
    // Forward: Q_raw (B, T, D) -> view (B, T, H, dk) -> transpose -> (B, H, T, dk) = Q_heads
    // Backward: dQ_heads (B, H, T, dk) -> transpose_12 -> (B, T, H, dk) -> view (B, T, D) = dQ_raw

    Tensor dQ_raw({B, T, D});
    Tensor dK_raw({B, T, D});
    Tensor dV_raw({B, T, D});

    {
        // Transpose backward: (B, H, T, dk) -> (B, T, H, dk)
        // The output of transpose_12 has shape derived from input dims swapped.
        // We want output (B, T, H, dk). Pass dQ_raw with shape (B, T, H, dk),
        // mutate after.
        dQ_raw.shape = {B, T, H, d_k};
        transpose_12(dQ_raw, dQ_heads);
        dQ_raw.shape = {B, T, D};   // re-view as (B, T, D)

        dK_raw.shape = {B, T, H, d_k};
        transpose_12(dK_raw, dK_heads);
        dK_raw.shape = {B, T, D};

        dV_raw.shape = {B, T, H, d_k};
        transpose_12(dV_raw, dV_heads);
        dV_raw.shape = {B, T, D};
    }

    // ===== Step 5: backward through input projections =====
    // Q_raw = x @ W_Q + b_Q
    // -> dW_Q = x^T @ dQ_raw
    //    db_Q = sum(dQ_raw, axis=batch)
    //    contribution to dx = dQ_raw @ W_Q^T

    // x is needed three times (one for each of Q, K, V projections).
    // Each linear_backward produces a separate "dx" contribution; we need
    // to sum all three.

    Tensor dx_from_Q({B, T, D});
    Tensor dx_from_K({B, T, D});
    Tensor dx_from_V({B, T, D});

    {
        Tensor& x_nc = const_cast<Tensor&>(x);
        to2d(x_nc, B * T, D);
        to2d(dQ_raw, B * T, D);
        to2d(dK_raw, B * T, D);
        to2d(dV_raw, B * T, D);
        to2d(dx_from_Q, B * T, D);
        to2d(dx_from_K, B * T, D);
        to2d(dx_from_V, B * T, D);

        linear_backward(dx_from_Q, dW_Q, db_Q, x_nc, W_Q, dQ_raw);
        linear_backward(dx_from_K, dW_K, db_K, x_nc, W_K, dK_raw);
        linear_backward(dx_from_V, dW_V, db_V, x_nc, W_V, dV_raw);

        to3d(x_nc, B, T, D);
        // dQ_raw, dK_raw, dV_raw are scratch — don't bother restoring
        to3d(dx_from_Q, B, T, D);
        to3d(dx_from_K, B, T, D);
        to3d(dx_from_V, B, T, D);
    }

    // ===== Step 6: sum the three contributions to dx =====
    // dx = dx_from_Q + dx_from_K + dx_from_V
    {
        int n = (int)dx.numel;
        auto h_Q = dx_from_Q.to_host();
        auto h_K = dx_from_K.to_host();
        auto h_V = dx_from_V.to_host();
        std::vector<float> out(n);
        for (int i = 0; i < n; ++i) out[i] = h_Q[i] + h_K[i] + h_V[i];
        dx.from_host(out);
    }
}