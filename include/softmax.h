#pragma once
#include "tensor.h"

void softmax_forward(Tensor& y, const Tensor& x);

void causal_softmax(Tensor& P, const Tensor& S);

void softmax_backward_rowwise(
    Tensor& dS, const Tensor& P, const Tensor& dP);