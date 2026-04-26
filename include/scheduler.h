#pragma once
#include <cmath>

inline constexpr float kPi = 3.14159265358979323846f;

inline float warmup_cosine_lr(
    int step, int warmup, int total,
    float lr_max, float lr_min)
{
    if (step < warmup) {
        if (warmup == 0) return lr_max;
        return lr_max * ((float)step / (float)warmup);
    }
    if (step >= total) return lr_min;

    float progress = (float)(step - warmup) / (float)(total - warmup);
    float cos_part = 0.5f * (1.0f + std::cos(kPi * progress));
    return lr_min + (lr_max - lr_min) * cos_part;
}