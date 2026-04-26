#include "scheduler.h"
#include <cstdio>

int main() {
    int total = 1000;
    int warmup = 100;
    float lr_max = 1e-3f;
    float lr_min = 1e-4f;

    std::printf("=== LR schedule sanity check ===\n");
    std::printf("warmup=%d, total=%d, lr_max=%.4f, lr_min=%.4f\n\n",
                warmup, total, lr_max, lr_min);

    // Print every 50 steps
    for (int step = 0; step <= total; step += 50) {
        float lr = warmup_cosine_lr(step, warmup, total, lr_max, lr_min);
        // Visualize as a bar
        int bars = (int)(50.f * lr / lr_max);
        std::printf("  step %4d  lr=%.5f  ", step, lr);
        for (int i = 0; i < bars; ++i) std::putchar('#');
        std::putchar('\n');
    }

    // Edge cases
    std::printf("\nEdge cases:\n");
    std::printf("  step 0:    lr = %.5f (should be 0)\n",
                warmup_cosine_lr(0, warmup, total, lr_max, lr_min));
    std::printf("  step %d:  lr = %.5f (should be %.5f)\n",
                warmup, warmup_cosine_lr(warmup, warmup, total, lr_max, lr_min), lr_max);
    std::printf("  step %d: lr = %.5f (should be %.5f)\n",
                total, warmup_cosine_lr(total, warmup, total, lr_max, lr_min), lr_min);
    std::printf("  step %d: lr = %.5f (should be %.5f)\n",
                total + 100, warmup_cosine_lr(total + 100, warmup, total, lr_max, lr_min), lr_min);

    return 0;
}