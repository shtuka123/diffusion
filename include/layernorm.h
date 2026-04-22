#pragma once
#include "tensor.h"

void layernorm_forward(
    Tensor& y, Tensor& rstd,
    const Tensor& x,
    const Tensor& gamma, const Tensor& beta,
    float eps = 1e-5f);

void layernorm_backward(
    Tensor& dx, Tensor& dgamma, Tensor& dbeta,
    const Tensor& dy, const Tensor& x,
    const Tensor& gamma, const Tensor& rstd);