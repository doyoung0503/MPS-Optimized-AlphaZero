cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CppAlphaZero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C++
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 17)

# Find LibTorch
set(Torch_DIR "/Users/doyoung/miniforge3/lib/python3.12/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find Foundation framework for @autoreleasepool
find_library(FOUNDATION_LIBRARY Foundation)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Define source files for Main App
set(MAIN_SOURCES 
    src/main.cpp 
    src/Chess.cpp 
    src/MCTS.mm 
    src/Trainer.mm
)

# Define source files for Grid Search
set(GRID_SOURCES 
    src/GridSearch.cpp 
    src/Chess.cpp 
    src/MCTS.mm 
    src/Trainer.mm
)

# Create executable: alphazero
add_executable(alphazero ${MAIN_SOURCES})
target_link_libraries(alphazero "${TORCH_LIBRARIES}" ${FOUNDATION_LIBRARY})
set_property(TARGET alphazero PROPERTY CXX_STANDARD 17)

# Create executable: grid_search
add_executable(grid_search ${GRID_SOURCES})
target_link_libraries(grid_search "${TORCH_LIBRARIES}" ${FOUNDATION_LIBRARY})
set_property(TARGET grid_search PROPERTY CXX_STANDARD 17)

# Create executable: test_random
add_executable(test_random src/TestRandom.cpp src/Chess.cpp src/MCTS.mm)
target_link_libraries(test_random "${TORCH_LIBRARIES}" ${FOUNDATION_LIBRARY})
set_property(TARGET test_random PROPERTY CXX_STANDARD 17)

# Compiler flags for optimization
if(MSVC)
  target_compile_options(alphazero PUBLIC /O2)
  target_compile_options(grid_search PUBLIC /O2)
else()
  target_compile_options(alphazero PUBLIC -O3 -march=native)
  target_compile_options(grid_search PUBLIC -O3 -march=native)
  target_compile_options(test_random PUBLIC -O3 -march=native)
endif()
