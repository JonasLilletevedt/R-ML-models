cmake_minimum_required(VERSION 3.16)

project(linear_regression_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------
# Build type and optimization defaults
# --------------------------------------------------------------------

# If the user did not specify a build type, default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Add aggressive optimization flags for Release on GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG"
        CACHE STRING "Release flags" FORCE
    )
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# --------------------------------------------------------------------
# OpenMP toggle
# --------------------------------------------------------------------

option(LINEAR_REGRESSION_ENABLE_OPENMP "Enable OpenMP parallelism" ON)

if(LINEAR_REGRESSION_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# --------------------------------------------------------------------
# Library
# --------------------------------------------------------------------

add_library(linear_regression_cpp SHARED
    src/linear_regression_v21.cpp
    src/linear_regression_v22.cpp
    src/linear_regression_c_api.cpp
)

target_include_directories(linear_regression_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(LINEAR_REGRESSION_ENABLE_OPENMP)
    target_link_libraries(linear_regression_cpp
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()
