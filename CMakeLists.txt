cmake_minimum_required (VERSION 2.6)

project (heterogeneous_openmp_parallel_for_each)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(./)

set(HEADER_FILES parallel_for_each.hpp)

add_executable(parallel_for_each main.cpp sample_class.cpp ${HEADER_FILES})

set(CMAKE_CXX_FLAGS -fopenmp)