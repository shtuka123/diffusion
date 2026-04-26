#include "tensor.h"
#include "tensor_io.h"
#include "diffusion.h"
#include "noise_schedule.h"
#include "cuda_utils.h"
#include <cstdio>
#include <fstream>
#include <vector>

static std::vector<int> load_int_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "can't open %s\n", path.c_str()); std::abort(); }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int> out(bytes / sizeof(int));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

int main() {
    std::printf("=== Noise schedule + q_sample vs PyTorch ===\n");

    NoiseSchedule sched;
    sched.build(1000, 1e-4f, 2e-2f);

    int fails = 0;

    // Verify schedule arrays
    Tensor expected_betas = load_tensor("data/ref/diff_betas.bin");
    float err_b = max_abs_diff(sched.beta, expected_betas);
    std::printf("  betas        err: %.3e  %s\n",
                err_b, err_b < 1e-6f ? "PASS" : "FAIL");
    if (err_b >= 1e-6f) ++fails;

    Tensor expected_ab = load_tensor("data/ref/diff_alpha_bar.bin");
    float err_ab = max_abs_diff(sched.alpha_bar, expected_ab);
    std::printf("  alpha_bar    err: %.3e  %s\n",
                err_ab, err_ab < 1e-5f ? "PASS" : "FAIL");
    if (err_ab >= 1e-5f) ++fails;

    // Sanity check on alpha_bar values
    auto h_ab = sched.alpha_bar.to_host();
    std::printf("  alpha_bar[0]   = %.8f\n", h_ab[0]);
    std::printf("  alpha_bar[T-1] = %.8f\n", h_ab[h_ab.size() - 1]);

    // q_sample test
    Tensor x0  = load_tensor("data/ref/diff_x0.bin");
    Tensor eps = load_tensor("data/ref/diff_eps.bin");
    Tensor expected_xt = load_tensor("data/ref/diff_xt.bin");

    auto h_ts = load_int_bin("data/ref/diff_timesteps.bin");
    int B = x0.size(0);
    if ((int)h_ts.size() != B) {
        std::fprintf(stderr, "timesteps count mismatch\n"); return 1;
    }

    int* d_ts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ts, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ts, h_ts.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    Tensor xt(x0.shape);
    q_sample(xt, x0, eps, d_ts, sched);

    float err_xt = max_abs_diff(xt, expected_xt);
    std::printf("  q_sample xt  err: %.3e  %s\n",
                err_xt, err_xt < 1e-5f ? "PASS" : "FAIL");
    if (err_xt >= 1e-5f) ++fails;

    cudaFree(d_ts);

    if (fails == 0) {
        std::printf("\nAll diffusion math tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}