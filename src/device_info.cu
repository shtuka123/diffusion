#include <cstdio>
#include "cuda_utils.h"

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("Found %d CUDA device(s)\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::printf("Device %d: %s\n", i, prop.name);
        std::printf("  Compute capability:   %d.%d\n", prop.major, prop.minor);
        std::printf("  SM count:             %d\n", prop.multiProcessorCount);
        std::printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        std::printf("  Warp size:            %d\n", prop.warpSize);
        std::printf("  Global memory:        %.2f GB\n",
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  Shared mem per block: %zu KB\n",
                    prop.sharedMemPerBlock / 1024);
        std::printf("\n");
    }
    return 0;
}