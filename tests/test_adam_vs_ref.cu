#include "tensor.h"
#include "tensor_io.h"
#include "parameter.h"
#include "optim.h"
#include "cuda_utils.h"
#include <cstdio>
#include <string>

int main() {
    std::printf("=== Adam optimizer vs PyTorch ===\n");

    Tensor w_init = load_tensor("data/ref/adam_w_init.bin");
    int n = (int)w_init.numel;

    // Build a Parameter with identical initial weights
    Parameter p({n});
    p.w.copy_from(w_init);

    AdamState state;
    state.lr = 1e-3f;
    state.beta1 = 0.9f;
    state.beta2 = 0.999f;
    state.eps = 1e-8f;

    int fails = 0;
    for (int step = 0; step < 5; ++step) {
        Tensor g = load_tensor("data/ref/adam_grad_" + std::to_string(step) + ".bin");
        p.grad.copy_from(g);

        // Single-parameter step using the all-params helper to share step++.
        std::vector<Parameter*> ps = {&p};
        adam_step_all(ps, state);

        Tensor w_expected = load_tensor("data/ref/adam_w_" + std::to_string(step + 1) + ".bin");
        float err = max_abs_diff(p.w, w_expected);
        bool pass = err < 1e-5f;
        std::printf("  step %d  max abs err: %.3e  %s\n",
                    step + 1, err, pass ? "PASS" : "FAIL");
        if (!pass) ++fails;
    }

    if (fails == 0) {
        std::printf("\nAdam matches PyTorch.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}