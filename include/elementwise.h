#pragma once
#include "tensor.h"

void relu_forward(Tensor& y, const Tensor& x);
void bias_add(Tensor& y, const Tensor& x, const Tensor& bias);
void relu_backward(Tensor& dx, const Tensor& dy, const Tensor& x);