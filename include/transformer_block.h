#pragma once
#include "tensor.h"
#include "mha.h"
#include "mlp.h"

struct TransformerBlockCache {
    Tensor t1;
    Tensor rstd1;
    Tensor a;
    Tensor r1;
    Tensor t2;
    Tensor rstd2;
    Tensor m;
    MHACache mha_cache;
    MLPCache mlp_cache;
};

void transformer_block_forward(
    Tensor& y,
    TransformerBlockCache& cache,
    const Tensor& x,
    const Tensor& gamma1, const Tensor& beta1,
    const Tensor& W_Q, const Tensor& b_Q,
    const Tensor& W_K, const Tensor& b_K,
    const Tensor& W_V, const Tensor& b_V,
    const Tensor& W_O, const Tensor& b_O,
    const Tensor& gamma2, const Tensor& beta2,
    const Tensor& W1_mlp, const Tensor& b1_mlp,
    const Tensor& W2_mlp, const Tensor& b2_mlp,
    int n_heads);

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
    int n_heads);