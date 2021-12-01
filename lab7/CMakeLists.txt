cmake_minimum_required(VERSION 3.20)
project(lab7)

set(CMAKE_CXX_STANDARD 17)

#add_executable(lab7 main.cpp matrix.cpp matrix.h matrixSimd.cpp matrixSimd.h)
add_executable(lab7 main.cpp matrix.cpp matrix.h matrixSSE.cpp matrixSSE.h)

#