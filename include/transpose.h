#pragma once
#include "tensor.h"

// Transpose dims 1 and 2 of a 4D tensor:
// (d0, d1, d2, d3) -> (d0, d2, d1, d3)
// Used for attention: (B, T, H, dk) <-> (B, H, T, dk).
void transpose_12(Tensor& y, const Tensor& x);