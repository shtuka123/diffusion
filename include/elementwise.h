#pragma once
#include "tensor.h"

void relu_forward(Tensor& y, const Tensor& x);
void bias_add(Tensor& y, const Tensor& x, const Tensor& bias);
void relu_backward(Tensor& dx, const Tensor& dy, const Tensor& x);
void argmax_row(int* preds_device, const Tensor& logits);
void gelu_forward(Tensor& y, const Tensor& x);
void gelu_backward(Tensor& dx, const Tensor& dy, const Tensor& x);

void add(Tensor& y, const Tensor& a, const Tensor& b);