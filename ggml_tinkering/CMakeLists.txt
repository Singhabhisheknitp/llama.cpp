# Minimum CMake version
cmake_minimum_required(VERSION 3.10)
# Project name
project(ggml_test)

# Add executable
add_executable(ggmltest ggmlt.c)

# Setting variables
set(GGML_BUILD_DIR ${CMAKE_SOURCE_DIR}/../build/ggml/src)
set(GGML_INCLUDE_DIR1 ${CMAKE_SOURCE_DIR}/../ggml/include)
set(GGML_INCLUDE_DIR2 ${CMAKE_SOURCE_DIR}/../ggml/src)

# (equivalent to -L./directory path)
link_directories(${GGML_BUILD_DIR})  

# Link libraries equivalent to (-lggml-base -lggml-cpu)
target_link_libraries(ggmltest 
${GGML_BUILD_DIR}/libggml-base.so
${GGML_BUILD_DIR}/libggml-cpu.so
)

# (equiavalent to -I ../ ggml/include)
target_include_directories(ggmltest PRIVATE 
    ${GGML_INCLUDE_DIR1}
    ${GGML_INCLUDE_DIR2}
)
