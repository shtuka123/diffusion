#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wraps CUDA runtime API calls. Aborts with file/line/function on error.
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",         \
                     __FILE__, __LINE__, #call,                         \
                     cudaGetErrorString(err));                          \
        std::abort();                                                   \
    }                                                                   \
} while (0)

// Kernel launches don't return a cudaError_t directly. Call this
// AFTER a kernel launch to check for errors. It also synchronizes,
// which means any async kernel errors surface immediately.
#define CUDA_CHECK_KERNEL() do {                                        \
    CUDA_CHECK(cudaGetLastError());                                     \
    CUDA_CHECK(cudaDeviceSynchronize());                                \
} while (0)

// Simple GPU-side timer using CUDA events. Events are recorded into
// the stream and timed by the GPU itself, which is more accurate than
// wall-clock timing for async work.
class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    // Non-copyable (owns CUDA event handles)
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Call after the work you want to measure. Returns elapsed ms.
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};