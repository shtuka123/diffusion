cmake_minimum_required(VERSION 3.18)
project(dit_from_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(hello src/hello.cu)
add_executable(device_info src/device_info.cu)
add_executable(test_cuda_check src/test_cuda_check.cu)
add_executable(test_tensor src/test_tensor.cu)