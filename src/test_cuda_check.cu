#include "cuda_utils.h"
#include <cstdio>

int main() {
    std::printf("About to do something bad...\n");

    // Try to allocate a ridiculous amount of memory.
    // size_t is unsigned, so -1 wraps to the maximum value.
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t)-1));

    std::printf("This line should never print.\n");
    return 0;
}