#pragma once
#include "parameter.h"
#include <cmath>

// SGD (existing)
void sgd_step(Parameter& p, float lr);
void sgd_step_all(const std::vector<Parameter*>& params, float lr);

// Adam
struct AdamState {
    int step = 0;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};

void adam_step(Parameter& p, AdamState& state);
void adam_step_all(const std::vector<Parameter*>& params, AdamState& state);