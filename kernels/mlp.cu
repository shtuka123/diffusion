#include "cuda_utils.h"
#include "tensor.h"
#include "linear.h"
#include "elementwise.h"
#include "matmul.h"

#include <cstdio>

// Transformer-style MLP sub-block: y = Linear(D, D)( GELU( Linear(D, 4D)(x) ) )
//
// Shapes (canonical):
//   x:   (B, T, D)
//   y:   (B, T, D)
//   W1:  (D, 4D), b1: (4D,)
//   W2:  (4D, D), b2: (D,)
//
// Cache (caller-owned, for reuse and backward):
//   h_pre:  (B, T, 4D)  output of first linear, pre-GELU
//   h_post: (B, T, 4D)  output of GELU, input to second linear
struct MLPCache {
    Tensor h_pre;
    Tensor h_post;
};

// Helper: 3D linear, treats (B, T, D_in) as (B*T, D_in).
// Same trick we used in mha.cu — call matmul_raw + bias_add directly.
static void linear_3d(Tensor& y, const Tensor& x,
                      const Tensor& W, const Tensor& b)
{
    int B = x.size(0), T = x.size(1);
    int D_in = x.size(2), D_out = W.size(1);
    matmul_raw(y.data, x.data, W.data, B * T, D_in, D_out);

    // bias_add as a flat vector — bias_add_kernel uses i % feat_dim, so
    // we can pass the (B*T, D_out) totals directly.
    extern __global__ void bias_add_kernel(const float*, const float*, float*,
                                           int, int);
    int n = B * T * D_out;
    int blk = 256, grd = (n + blk - 1) / blk;
    bias_add_kernel<<<grd, blk>>>(y.data, b.data, y.data, n, D_out);
    CUDA_CHECK_KERNEL();
}

void mlp_forward(
    Tensor& y,
    MLPCache& cache,
    const Tensor& x,
    const Tensor& W1, const Tensor& b1,
    const Tensor& W2, const Tensor& b2)
{
    // h_pre = Linear_1(x)
    linear_3d(cache.h_pre, x, W1, b1);
    // h_post = GELU(h_pre)
    gelu_forward(cache.h_post, cache.h_pre);
    // y = Linear_2(h_post)
    linear_3d(y, cache.h_post, W2, b2);
}

// Backward.
// Inputs:
//   dy:    (B, T, D)  upstream gradient
//   x:     (B, T, D)  forward input (cached)
//   cache: contains h_pre, h_post
//   W1, W2: weight matrices
// Outputs:
//   dx:    (B, T, D)
//   dW1:   (D, 4D), db1: (4D,)
//   dW2:   (4D, D), db2: (D,)
//
// We'll re-use linear_backward, which expects 2D inputs.
// Strategy: temporarily mutate shapes (numel preserved) to call linear_backward,
// then restore.
void mlp_backward(
    Tensor& dx,
    Tensor& dW1, Tensor& db1,
    Tensor& dW2, Tensor& db2,
    const Tensor& dy,
    const Tensor& x,
    const MLPCache& cache,
    const Tensor& W1, const Tensor& W2)
{
    int B = x.size(0), T = x.size(1), D = x.size(2);
    int D_hidden = W1.size(1);  // = 4D conventionally

    // Shape-mutate to 2D for linear_backward. We work on non-const aliases
    // by const_cast — these are just shape mutations, numel is preserved.
    auto to2d_3to2 = [](Tensor& t, int rows, int cols) {
        t.shape = {rows, cols};
    };
    auto restore_3d = [](Tensor& t, int B, int T, int D) {
        t.shape = {B, T, D};
    };

    // Step 1: backward through Linear_2.
    //   Inputs:  h_post (B, T, 4D), W2 (4D, D), dy (B, T, D)
    //   Outputs: dh_post (B, T, 4D), dW2 (4D, D), db2 (D,)
    Tensor dh_post(cache.h_post.shape);

    // Mutate shapes to 2D for linear_backward's 2D-only API.
    Tensor& h_post_nc = const_cast<Tensor&>(cache.h_post);
    Tensor& dy_nc     = const_cast<Tensor&>(dy);
    to2d_3to2(h_post_nc, B * T, D_hidden);
    to2d_3to2(dy_nc, B * T, D);
    to2d_3to2(dh_post, B * T, D_hidden);

    linear_backward(dh_post, dW2, db2, h_post_nc, W2, dy_nc);

    // Restore
    restore_3d(h_post_nc, B, T, D_hidden);
    restore_3d(dy_nc, B, T, D);
    restore_3d(dh_post, B, T, D_hidden);

    // Step 2: backward through GELU.
    //   dh_pre = gelu_backward(dh_post, h_pre)
    Tensor dh_pre(cache.h_pre.shape);
    gelu_backward(dh_pre, dh_post, cache.h_pre);

    // Step 3: backward through Linear_1.
    //   Inputs:  x (B, T, D), W1 (D, 4D), dh_pre (B, T, 4D)
    //   Outputs: dx (B, T, D), dW1 (D, 4D), db1 (4D,)
    Tensor& x_nc = const_cast<Tensor&>(x);
    to2d_3to2(x_nc, B * T, D);
    to2d_3to2(dh_pre, B * T, D_hidden);
    to2d_3to2(dx, B * T, D);

    linear_backward(dx, dW1, db1, x_nc, W1, dh_pre);

    restore_3d(x_nc, B, T, D);
    restore_3d(dh_pre, B, T, D_hidden);
    restore_3d(dx, B, T, D);
}