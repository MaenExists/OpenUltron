cmake_minimum_required(VERSION 3.16)
project(OpenUltron
    VERSION 0.1.0
    DESCRIPTION "Sparse Predictive Adaptive Recurrent Kernel (SPARK)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

include_directories(include)

# Core library
set(SPARK_CORE_SOURCES
    src/graph.cpp
    src/network.cpp
    src/metrics.cpp
    src/environment.cpp
    src/learning.cpp
    src/trainer.cpp
    src/baselines.cpp
    src/experiment_runner.cpp
    src/local_learning.cpp
)

add_library(spark_core STATIC ${SPARK_CORE_SOURCES})
target_include_directories(spark_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main executable
add_executable(spark src/main.cpp)
target_link_libraries(spark PRIVATE spark_core)

# Provide openultron symlink/alias to spark
add_custom_command(TARGET spark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:spark> $<TARGET_FILE_DIR:spark>/openultron
    COMMENT "Creating openultron symlink to spark"
)

# Testing
enable_testing()

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph PRIVATE spark_core)
add_test(NAME test_graph COMMAND test_graph)

add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE spark_core)
add_test(NAME test_forward COMMAND test_forward)

add_executable(test_gradients tests/test_gradients.cpp)
target_link_libraries(test_gradients PRIVATE spark_core)
add_test(NAME test_gradients COMMAND test_gradients)

add_executable(test_learning tests/test_learning.cpp)
target_link_libraries(test_learning PRIVATE spark_core)
add_test(NAME test_learning COMMAND test_learning)

add_executable(test_baselines tests/test_baselines.cpp)
target_link_libraries(test_baselines PRIVATE spark_core)
add_test(NAME test_baselines COMMAND test_baselines)

add_executable(test_regression tests/test_regression.cpp)
target_link_libraries(test_regression PRIVATE spark_core)
add_test(NAME test_regression COMMAND test_regression)

add_executable(test_local_learning tests/test_local_learning.cpp)
target_link_libraries(test_local_learning PRIVATE spark_core)
add_test(NAME test_local_learning COMMAND test_local_learning)
