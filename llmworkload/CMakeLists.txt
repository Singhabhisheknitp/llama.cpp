# Minimum CMake version
cmake_minimum_required(VERSION 3.10)
# Project name
project(ggml_test)

# Add executable
add_executable(gg ggmlt.c)

# Setting variables
set(GGML_BUILD_DIR ${CMAKE_SOURCE_DIR}/../build/ggml/src)
set(GGML_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../ggml/include)

# (equivalent to -L./directory path)
link_directories(${GGML_BUILD_DIR})  

# Link libraries equivalent to (-lggml-base -lggml-cpu)
target_link_libraries(gg 
${GGML_BUILD_DIR}/libggml-base.so
${GGML_BUILD_DIR}/libggml-cpu.so
)

# (equiavalent to -I ../ ggml/include)
target_include_directories(gg PRIVATE 
    ${GGML_INCLUDE_DIR}
)