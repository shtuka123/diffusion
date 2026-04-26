#pragma once
#include "tensor.h"

void sinusoidal_embed(
    Tensor& emb,
    const int* timesteps_device, int B,
    float max_period = 10000.0f);