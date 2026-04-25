#pragma once
#include "tensor.h"

void embedding_forward(
    Tensor& y, const int* tokens_device,
    const Tensor& E, int B, int T);

void embedding_backward(
    Tensor& dE,
    const Tensor& dy, const int* tokens_device,
    int B, int T);

void positional_add_forward(
    Tensor& y, const Tensor& P, int B, int T);

void positional_add_backward(
    Tensor& dP, const Tensor& dy, int B, int T);