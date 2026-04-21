#pragma once
#include "tensor.h"

void linear_forward(Tensor& y,
                    const Tensor& x,
                    const Tensor& W,
                    const Tensor& b);

void linear_backward(Tensor& dx, Tensor& dW, Tensor& db,
                     const Tensor& x, const Tensor& W, const Tensor& dy);