cmake_minimum_required(VERSION 3.18)
project(dit_from_scratch LANGUAGES CXX CUDA)

# ===================================================================
# Standards and build options
# ===================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 40-series (Ada Lovelace)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ===================================================================
# Headers and CUDA runtime
# ===================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ===================================================================
# Kernel source groups
# Each kernel gets a variable; targets reference these by name.
# Adding a new kernel: add one set() here, use in add_target() below.
# ===================================================================
set(KERNEL_MATMUL         kernels/matmul.cu)
set(KERNEL_MATMUL_CUBLAS  kernels/matmul_cublas.cu)
set(KERNEL_MATMUL_CPU     kernels/matmul_cpu.cpp)
set(KERNEL_ELEMENTWISE    kernels/elementwise.cu)
set(KERNEL_SOFTMAX        kernels/softmax.cu)
set(KERNEL_CE             kernels/cross_entropy.cu)
set(KERNEL_LINEAR         kernels/linear.cu)
set(KERNEL_OPTIM          kernels/optim.cu)
# ===================================================================
# Helper: declare an executable target
#
# Usage:
#   add_target(name "main.cu" kernel_var_1 kernel_var_2 ...)
#
# The kernel args should be variable values (e.g. ${KERNEL_MATMUL}),
# not literal filenames. This keeps filenames defined in exactly one place.
# ===================================================================
function(add_target name main_src)
  add_executable(${name} ${main_src} ${ARGN})
endfunction()

# ===================================================================
# Utility tools — no kernel dependencies
# ===================================================================
add_target(hello             src/hello.cu)
add_target(device_info       src/device_info.cu)
add_target(test_cuda_check   src/test_cuda_check.cu)
add_target(test_tensor       src/test_tensor.cu)

# ===================================================================
# Forward-pass validation tests (vs PyTorch reference)
# ===================================================================
add_target(test_matmul              tests/test_matmul.cu              ${KERNEL_MATMUL})
add_target(test_matmul_vs_ref       tests/test_matmul_vs_ref.cu       ${KERNEL_MATMUL})
add_target(test_elementwise_vs_ref  tests/test_elementwise_vs_ref.cu  ${KERNEL_ELEMENTWISE})
add_target(test_softmax_vs_ref      tests/test_softmax_vs_ref.cu      ${KERNEL_SOFTMAX})
add_target(test_ce_vs_ref           tests/test_ce_vs_ref.cu           ${KERNEL_CE})

# ===================================================================
# Backward-pass tests (grad-checked)
# ===================================================================
add_target(test_relu_backward       tests/test_relu_backward.cu       ${KERNEL_ELEMENTWISE})
add_target(test_ce_backward         tests/test_ce_backward.cu         ${KERNEL_CE})

# ===================================================================
# Benchmarks
# ===================================================================
add_target(bench_matmul
    tests/bench_matmul.cu
    ${KERNEL_MATMUL} ${KERNEL_MATMUL_CUBLAS} ${KERNEL_MATMUL_CPU})
target_link_libraries(bench_matmul PRIVATE cublas)

add_target(bench_elementwise
    tests/bench_elementwise.cu
    ${KERNEL_ELEMENTWISE})

add_target(test_linear_vs_ref
    tests/test_linear_vs_ref.cu
    ${KERNEL_LINEAR} ${KERNEL_MATMUL} ${KERNEL_ELEMENTWISE})

add_target(test_linear_backward
    tests/test_linear_backward.cu
    ${KERNEL_LINEAR} ${KERNEL_MATMUL} ${KERNEL_ELEMENTWISE})

add_target(test_optim_step
    tests/test_optim_step.cu
    ${KERNEL_LINEAR} ${KERNEL_MATMUL} ${KERNEL_ELEMENTWISE}
    ${KERNEL_CE} ${KERNEL_OPTIM})

add_target(test_mnist_loader
    tests/test_mnist_loader.cu)

# ===================================================================
# Convenience meta-targets
# Run a logical group with:
#   cmake --build build --target validation_tests
#   cmake --build build --target backward_tests
#   cmake --build build --target benchmarks
# ===================================================================
add_custom_target(validation_tests DEPENDS
    test_tensor
    test_matmul
    test_matmul_vs_ref
    test_elementwise_vs_ref
    test_softmax_vs_ref
    test_ce_vs_ref
    test_linear_vs_ref)   # new

add_custom_target(backward_tests DEPENDS
    test_relu_backward
    test_ce_backward
    test_linear_backward)  # new

add_custom_target(benchmarks DEPENDS
    bench_matmul
    bench_elementwise)

