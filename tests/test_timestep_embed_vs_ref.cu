#include "tensor.h"
#include "tensor_io.h"
#include "timestep_embed.h"
#include "elementwise.h"
#include "grad_check.h"
#include "cuda_utils.h"

#include <cstdio>
#include <vector>
#include <fstream>

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

__global__ void sum_squares_kernel(const float* y, float* loss, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += y[i] * y[i];
    *loss = s;
}

int main() {
    int fails = 0;

    // ===== Sinusoidal embedding =====
    std::printf("=== Sinusoidal timestep embedding ===\n");
    auto h_ts = load_int_bin("data/ref/temb_timesteps.bin");
    Tensor expected_emb = load_tensor("data/ref/temb_emb.bin");
    int B = (int)h_ts.size();
    int d = expected_emb.size(1);

    int* d_ts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ts, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ts, h_ts.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    Tensor emb({B, d});
    sinusoidal_embed(emb, d_ts, B);

    float err_emb = max_abs_diff(emb, expected_emb);
    std::printf("  B=%d, d=%d\n", B, d);
    std::printf("  max abs err: %.3e  %s\n",
                err_emb, err_emb < 1e-5f ? "PASS" : "FAIL");
    if (err_emb >= 1e-5f) ++fails;

    cudaFree(d_ts);

    // ===== SiLU forward + backward =====
    std::printf("\n=== SiLU forward + backward ===\n");
    Tensor x_silu       = load_tensor("data/ref/silu_x.bin");
    Tensor expected_y   = load_tensor("data/ref/silu_y.bin");
    Tensor dy_silu      = load_tensor("data/ref/silu_dy.bin");
    Tensor expected_dx  = load_tensor("data/ref/silu_dx.bin");

    Tensor y_silu(x_silu.shape);
    silu_forward(y_silu, x_silu);
    float err_y = max_abs_diff(y_silu, expected_y);
    std::printf("  forward max abs err:  %.3e  %s\n",
                err_y, err_y < 1e-5f ? "PASS" : "FAIL");
    if (err_y >= 1e-5f) ++fails;

    Tensor dx_silu(x_silu.shape);
    silu_backward(dx_silu, dy_silu, x_silu);
    float err_dx = max_abs_diff(dx_silu, expected_dx);
    std::printf("  backward max abs err: %.3e  %s\n",
                err_dx, err_dx < 1e-4f ? "PASS" : "FAIL");
    if (err_dx >= 1e-4f) ++fails;

    // Grad check on SiLU backward (tiny input)
    std::printf("\n=== SiLU backward: grad_check ===\n");
    Tensor x_c = Tensor::randn({2, 5}, 7);
    Tensor y_c({2, 5});
    Tensor loss({1});

    auto forward = [&](Tensor& x_in) -> float {
        silu_forward(y_c, x_in);
        sum_squares_kernel<<<1, 1>>>(y_c.data, loss.data, (int)y_c.numel);
        CUDA_CHECK_KERNEL();
        return loss.to_host()[0];
    };
    forward(x_c);
    auto y_host = y_c.to_host();
    std::vector<float> dy_host(y_c.numel);
    for (size_t i = 0; i < y_c.numel; ++i) dy_host[i] = 2.f * y_host[i];
    Tensor dy_c(y_c.shape); dy_c.from_host(dy_host);
    Tensor dx_c({2, 5});
    silu_backward(dx_c, dy_c, x_c);

    float rel = grad_check(forward, x_c, dx_c, 1e-2f, false);
    std::printf("  Max relative error: %.3e  %s\n",
                rel, rel < 1e-2f ? "PASS" : "FAIL");
    if (rel >= 1e-2f) ++fails;

    if (fails == 0) {
        std::printf("\nAll timestep + SiLU tests passed.\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n%d failed.\n", fails);
        return 1;
    }
}