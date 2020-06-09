
cmake_minimum_required (VERSION 2.6)

project (openmp_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(./)

set(HEADER_FILES mytime.h)

add_executable(openmp_example openmp_example.cpp ${HEADER_FILES})

add_executable(openmp_function_example openmp_function_example.cpp ${HEADER_FILES})


set(CMAKE_CXX_FLAGS -fopenmp)


