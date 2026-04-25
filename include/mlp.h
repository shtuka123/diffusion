#pragma once
#include "tensor.h"

struct MLPCache {
    Tensor h_pre;
    Tensor h_post;
};

void mlp_forward(
    Tensor& y,
    MLPCache& cache,
    const Tensor& x,
    const Tensor& W1, const Tensor& b1,
    const Tensor& W2, const Tensor& b2);

void mlp_backward(
    Tensor& dx,
    Tensor& dW1, Tensor& db1,
    Tensor& dW2, Tensor& db2,
    const Tensor& dy,
    const Tensor& x,
    const MLPCache& cache,
    const Tensor& W1, const Tensor& W2);