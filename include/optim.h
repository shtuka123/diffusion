#pragma once
#include "parameter.h"

void sgd_step(Parameter& p, float lr);
void sgd_step_all(const std::vector<Parameter*>& params, float lr);