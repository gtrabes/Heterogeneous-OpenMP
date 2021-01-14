cmake_minimum_required (VERSION 2.6)
message(STATUS "CMake version: ${CMAKE_VERSION}")

#if (NOT CMAKE_CUDA_COMPILER)
#    set(CMAKE_CUDA_COMPILER "${CMAKE_SOURCE_DIR}/usr/local/cuda/bin/nvcc")
#endif()

SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)

project (heterogeneous_computing)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--ptxas-options=-v" )

include_directories( ${CUDA_INCLUDE_DIRS} )

include_directories(./)

set(HEADER_FILES cpu_parallel_openmp.hpp gpu_parallel_cuda.hpp)

set(CUDA_FILES gpu_parallel_cuda.cu)

#set(HEADER_FILES cpu_parallel_openmp.hpp)


enable_language("CUDA")
add_executable(heterogeneous_openmp main.cpp sample_class.cpp ${HEADER_FILES} ${CUDA_FILES})

set(CMAKE_CXX_FLAGS -fopenmp)
