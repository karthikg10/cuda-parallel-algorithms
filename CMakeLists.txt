cmake_minimum_required(VERSION 3.18)
project(cuda_parallel_algorithms CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

find_package(CUDAToolkit REQUIRED)

add_executable(reduction_bench kernels/parallel_reduction.cu)
add_executable(histogram_bench  kernels/histogram.cu)
add_executable(scan_bench       kernels/prefix_scan.cu)
add_executable(sort_bench       kernels/bitonic_sort.cu)

foreach(target reduction_bench histogram_bench scan_bench sort_bench)
    target_link_libraries(${target} CUDA::cudart)
endforeach()
