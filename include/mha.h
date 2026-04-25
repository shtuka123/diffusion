#pragma once
#include "tensor.h"

struct MHACache {
    Tensor Q_heads, K_heads, V_heads;
    Tensor P;
    Tensor attn_out_flat;
    Tensor Q_raw, K_raw, V_raw;
    Tensor attn_out_heads;
    Tensor S_buf;
};

void mha_forward(
    Tensor& y,
    MHACache& cache,
    const Tensor& x,
    const Tensor& W_Q, const Tensor& b_Q,
    const Tensor& W_K, const Tensor& b_K,
    const Tensor& W_V, const Tensor& b_V,
    const Tensor& W_O, const Tensor& b_O,
    int n_heads);

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
    int n_heads);