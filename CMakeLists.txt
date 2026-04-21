cmake_minimum_required(VERSION 3.18)
project(dit_from_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
# Make CUDA's headers (cuda_runtime.h, etc.) findable from .cpp files too,
# not just .cu files. CMake's CUDA support only auto-adds them for nvcc.
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(hello src/hello.cu)
add_executable(device_info src/device_info.cu)
add_executable(test_cuda_check src/test_cuda_check.cu)
add_executable(test_tensor src/test_tensor.cu)

add_executable(test_matmul tests/test_matmul.cu kernels/matmul.cu)

# Benchmark — needs all three implementations and cuBLAS
add_executable(bench_matmul
    tests/bench_matmul.cu
    kernels/matmul.cu
    kernels/matmul_cublas.cu
    kernels/matmul_cpu.cpp)
target_link_libraries(bench_matmul PRIVATE cublas)