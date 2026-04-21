#pragma once
#include "tensor.h"

void cross_entropy_forward(
    Tensor& loss_scalar,
    Tensor& per_row_loss,
    const Tensor& logits,
    const int* labels_device,
    int B, int C);

void cross_entropy_backward(
    Tensor& dlogits,
    const Tensor& logits,
    const int* labels_device,
    int B, int C);