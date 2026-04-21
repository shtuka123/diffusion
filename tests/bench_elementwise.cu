#include "tensor.h"
#include "elementwise.h"
#include "cuda_utils.h"
#include <cstdio>

int main() {
    // Large enough to dominate launch overhead
    int N = 64 * 1024 * 1024;  // 64M floats = 256 MB
    Tensor x = Tensor::randn({N}, 42);
    Tensor y({N});

    // Warmup
    relu_forward(y, x);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time it
    CudaTimer t;
    t.start();
    for (int i = 0; i < 10; ++i) relu_forward(y, x);
    float ms = t.stop_ms() / 10.f;

    double bytes = 2.0 * N * sizeof(float);  // read + write
    double gbps = bytes / (ms / 1000.0) / 1e9;
    std::printf("ReLU on %d floats: %.3f ms, %.1f GB/s\n", N, ms, gbps);
    return 0;
}