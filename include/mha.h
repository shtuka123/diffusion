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