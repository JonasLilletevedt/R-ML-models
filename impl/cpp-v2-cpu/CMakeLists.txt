cmake_minimum_required(VERSION 3.16)

project(linear_regression_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LINEAR_REGRESSION_ENABLE_OPENMP "Enable OpenMP parallelism" ON)

if(LINEAR_REGRESSION_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

add_library(linear_regression_cpp SHARED
    src/linear_regression.cpp
    src/linear_regression_c_api.cpp)

target_include_directories(linear_regression_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(LINEAR_REGRESSION_ENABLE_OPENMP)
    target_link_libraries(linear_regression_cpp
        PRIVATE
            OpenMP::OpenMP_CXX)
endif()
