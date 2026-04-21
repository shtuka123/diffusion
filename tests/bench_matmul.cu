#include "matmul.h"
#include "tensor.h"
#include "cuda_utils.h"
#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

// Time a callable in milliseconds. Repeats N times, returns median.
template <typename Fn>
float median_time_ms(Fn fn, int repeats = 5) {
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        CudaTimer t;
        t.start();
        fn();
        times.push_back(t.stop_ms());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

// Time a CPU function with std::chrono.
template <typename Fn>
float cpu_time_ms(Fn fn, int repeats = 1) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return (float)us / 1000.f / (float)repeats;
}

// Compare two host vectors. Returns max abs error.
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

void run_size(int M, int K, int N) {
    std::printf("\n===== M=%d K=%d N=%d  =====\n", M, K, N);
    std::printf("           (FLOPs = 2 * M * K * N = %.2f GFLOP per call)\n",
                2.0 * M * K * N / 1e9);

    // Build inputs
    Tensor A = Tensor::randn({M, K}, 1);
    Tensor B = Tensor::randn({K, N}, 2);
    Tensor C_naive({M, N});
    Tensor C_cublas({M, N});

    auto h_A = A.to_host();
    auto h_B = B.to_host();
    std::vector<float> h_C_cpu(M * N);

    // ---- Time CPU (skip if huge) ----
    float t_cpu_ms = -1.f;
    if ((long long)M * K * N <= 200LL * 1000 * 1000) {  // ~200M FMAs cap
        t_cpu_ms = cpu_time_ms([&]{ matmul_cpu(h_C_cpu, h_A, h_B, M, K, N); });
    } else {
        std::printf("  (CPU skipped — too big)\n");
    }

    // ---- Time naive CUDA ----
    // warmup (first launch is slow)
    matmul(C_naive, A, B);
    CUDA_CHECK(cudaDeviceSynchronize());
    float t_naive_ms = median_time_ms([&]{
        matmul(C_naive, A, B);
    });

    // ---- Time cuBLAS ----
    matmul_cublas(C_cublas, A, B);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    float t_cublas_ms = median_time_ms([&]{
        matmul_cublas(C_cublas, A, B);
    });

    // ---- Correctness: all three should agree ----
    auto h_naive = C_naive.to_host();
    auto h_cublas = C_cublas.to_host();

    float err_naive_vs_cublas = max_abs_diff(h_naive, h_cublas);
    float err_cpu_vs_cublas = -1.f;
    if (t_cpu_ms > 0) err_cpu_vs_cublas = max_abs_diff(h_C_cpu, h_cublas);

    // ---- Report ----
    double gflops = 2.0 * M * K * N / 1e9;
    auto rate = [&](float ms) { return ms > 0 ? (float)(gflops / (ms / 1000.0)) : 0.f; };

    std::printf("\n  Implementation     |   Time (ms)  |   GFLOP/s   |  vs cuBLAS\n");
    std::printf("  -------------------+--------------+-------------+-----------\n");
    if (t_cpu_ms > 0) {
        std::printf("  CPU (naive)        | %12.3f | %11.2f | %8.1fx slower\n",
                    t_cpu_ms, rate(t_cpu_ms), t_cpu_ms / t_cublas_ms);
    }
    std::printf("  CUDA (naive)       | %12.3f | %11.2f | %8.1fx slower\n",
                t_naive_ms, rate(t_naive_ms), t_naive_ms / t_cublas_ms);
    std::printf("  cuBLAS             | %12.3f | %11.2f | %8s\n",
                t_cublas_ms, rate(t_cublas_ms), "1.0x");

    std::printf("\n  Correctness:\n");
    std::printf("    max|naive - cuBLAS|  = %.2e\n", err_naive_vs_cublas);
    if (err_cpu_vs_cublas >= 0)
        std::printf("    max|CPU   - cuBLAS|  = %.2e\n", err_cpu_vs_cublas);

    // For float32 accumulation, ~1e-3 absolute error at K~1000 is normal.
    // Different kernels accumulate in different order; small drift is expected.
    if (err_naive_vs_cublas > 1e-1f) {
        std::fprintf(stderr, "WARN: naive disagrees with cuBLAS by %.2e — likely a bug\n",
                     err_naive_vs_cublas);
    }
}

int main() {
    std::printf("Matmul benchmark — three implementations head-to-head\n");
    std::printf("======================================================\n");

    run_size(128, 128, 128);
    run_size(512, 512, 512);
    run_size(1024, 1024, 1024);
    run_size(2048, 2048, 2048);     // CPU skipped here automatically

    std::printf("\nDone.\n");
    return 0;
}