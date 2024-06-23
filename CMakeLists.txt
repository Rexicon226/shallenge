cmake_minimum_required(VERSION 3.10)
project(shallenge, CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/*.cu"
    "${CMAKE_SOURCE_DIR}/*.cuh"
)

# Create a custom target for formatting
add_custom_target(format
    COMMAND clang-format -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
)

add_executable(shallenge main.cu helper.cuh sha256.cuh)

target_link_libraries(shallenge)
