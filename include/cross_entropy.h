#pragma once
#include "tensor.h"

void cross_entropy_forward(
    Tensor& loss_scalar,
    Tensor& per_row_loss,
    const Tensor& logits,
    const int* labels_device,
    int B, int C);