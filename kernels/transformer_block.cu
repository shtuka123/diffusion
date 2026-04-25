#include "cuda_utils.h"
#include "tensor.h"
#include "layernorm.h"
#include "mha.h"
#include "mlp.h"
#include "elementwise.h"

#include <cstdio>

// Pre-LN Transformer block:
//
//   t1 = layernorm_1(x)
//   a  = mha(t1)
//   r1 = x + a
//   t2 = layernorm_2(r1)
//   m  = mlp(t2)
//   y  = r1 + m
//
// Cache for backward includes both LN's rstd, attention's P, MLP's intermediates.
struct TransformerBlockCache {
    Tensor t1;          // (B, T, D) — output of LN1
    Tensor rstd1;       // (B*T,)
    Tensor a;           // (B, T, D) — attention output
    Tensor r1;          // (B, T, D) — first residual
    Tensor t2;          // (B, T, D) — output of LN2
    Tensor rstd2;       // (B*T,)
    Tensor m;           // (B, T, D) — MLP output

    MHACache mha_cache;
    MLPCache mlp_cache;
};

void transformer_block_forward(
    Tensor& y,
    TransformerBlockCache& cache,
    const Tensor& x,
    // LN1 params
    const Tensor& gamma1, const Tensor& beta1,
    // MHA params
    const Tensor& W_Q, const Tensor& b_Q,
    const Tensor& W_K, const Tensor& b_K,
    const Tensor& W_V, const Tensor& b_V,
    const Tensor& W_O, const Tensor& b_O,
    // LN2 params
    const Tensor& gamma2, const Tensor& beta2,
    // MLP params
    const Tensor& W1_mlp, const Tensor& b1_mlp,
    const Tensor& W2_mlp, const Tensor& b2_mlp,
    int n_heads)
{
    // t1 = LN1(x)
    layernorm_forward(cache.t1, cache.rstd1, x, gamma1, beta1);

    // a = MHA(t1)
    mha_forward(cache.a, cache.mha_cache, cache.t1,
                W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
                n_heads);

    // r1 = x + a
    add(cache.r1, x, cache.a);

    // t2 = LN2(r1)
    layernorm_forward(cache.t2, cache.rstd2, cache.r1, gamma2, beta2);

    // m = MLP(t2)
    mlp_forward(cache.m, cache.mlp_cache, cache.t2,
                W1_mlp, b1_mlp, W2_mlp, b2_mlp);

    // y = r1 + m
    add(y, cache.r1, cache.m);
}

// Backward: traverse the forward graph in reverse.
//
//   y     = r1 + m       -> dr1 ← dy, dm ← dy
//   m     = MLP(t2)      -> dt2 ← MLP_backward(dm)
//   t2    = LN2(r1)      -> dr1 += LN2_backward(dt2)
//   r1    = x + a        -> dx ← dr1, da ← dr1
//   a     = MHA(t1)      -> dt1 ← MHA_backward(da)
//   t1    = LN1(x)       -> dx += LN1_backward(dt1)

void transformer_block_backward(
    Tensor& dx,
    Tensor& dgamma1, Tensor& dbeta1,
    Tensor& dW_Q, Tensor& db_Q,
    Tensor& dW_K, Tensor& db_K,
    Tensor& dW_V, Tensor& db_V,
    Tensor& dW_O, Tensor& db_O,
    Tensor& dgamma2, Tensor& dbeta2,
    Tensor& dW1_mlp, Tensor& db1_mlp,
    Tensor& dW2_mlp, Tensor& db2_mlp,
    const Tensor& dy,
    const Tensor& x,
    TransformerBlockCache& cache,
    const Tensor& gamma1,
    const Tensor& W_Q, const Tensor& W_K,
    const Tensor& W_V, const Tensor& W_O,
    const Tensor& gamma2,
    const Tensor& W1_mlp, const Tensor& W2_mlp,
    int n_heads)
{
    int B = x.size(0), T = x.size(1), D = x.size(2);

    // y = r1 + m: dy goes to both branches.
    // We'll accumulate dr1 from this (= dy) and the LN2 backward path.
    // For dm: just dy.
    // dr1 starts as a copy of dy.

    Tensor dr1(x.shape);
    dr1.copy_from(dy);   // dr1 = dy from r1 branch
    // dm = dy as well; we use dy directly when calling mlp_backward.

    // MLP backward: takes dy=dy as upstream, produces dt2 + dW1_mlp + db1_mlp + dW2_mlp + db2_mlp.
    Tensor dt2({B, T, D});
    mlp_backward(dt2, dW1_mlp, db1_mlp, dW2_mlp, db2_mlp,
                 dy, cache.t2, cache.mlp_cache,
                 W1_mlp, W2_mlp);

    // LN2 backward: dt2 -> contribution to dr1, plus dgamma2, dbeta2.
    Tensor dr1_from_LN2({B, T, D});
    layernorm_backward(dr1_from_LN2, dgamma2, dbeta2,
                       dt2, cache.r1, gamma2, cache.rstd2);

    // Accumulate: dr1 += dr1_from_LN2
    {
        auto h_a = dr1.to_host();
        auto h_b = dr1_from_LN2.to_host();
        std::vector<float> out(dr1.numel);
        for (size_t i = 0; i < dr1.numel; ++i) out[i] = h_a[i] + h_b[i];
        dr1.from_host(out);
    }

    // r1 = x + a: dr1 fans out to both. dx_partial1 = dr1; da = dr1.
    // We'll accumulate dx as we go; start with dr1's contribution.
    dx.copy_from(dr1);   // first contribution: from skip path

    // MHA backward: da=dr1 (treating it as the gradient at a) -> dt1 + all MHA gradients
    // We pass dr1 as the upstream gradient since da = dr1.
    Tensor dt1({B, T, D});
    mha_backward(dt1,
                 dW_Q, db_Q, dW_K, db_K, dW_V, db_V, dW_O, db_O,
                 dr1, cache.t1, cache.mha_cache,
                 W_Q, W_K, W_V, W_O, n_heads);

    // LN1 backward: dt1 -> contribution to dx, plus dgamma1, dbeta1.
    Tensor dx_from_LN1({B, T, D});
    layernorm_backward(dx_from_LN1, dgamma1, dbeta1,
                       dt1, x, gamma1, cache.rstd1);

    // dx += dx_from_LN1
    {
        auto h_a = dx.to_host();
        auto h_b = dx_from_LN1.to_host();
        std::vector<float> out(dx.numel);
        for (size_t i = 0; i < dx.numel; ++i) out[i] = h_a[i] + h_b[i];
        dx.from_host(out);
    }
}