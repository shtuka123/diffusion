#pragma once
#include "tensor.h"

void attention_forward(
    Tensor& Y,
    Tensor& P_out,            // softmax probs, saved for backward
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& S_buf);

void attention_backward(
    Tensor& dQ, Tensor& dK, Tensor& dV,
    const Tensor& dY,
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& P,
    Tensor& dP_buf, Tensor& dS_buf);