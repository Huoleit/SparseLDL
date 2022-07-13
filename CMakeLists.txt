cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(SparseLDL VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

add_library(${PROJECT_NAME}
  src/Types.cpp
)
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)
target_compile_options(${PROJECT_NAME} PUBLIC -Wfatal-errors -march=native -msse2 -mavx2 -mavx)

# ###############
# ## Benchmark ##
# ###############
add_subdirectory(benchmark)

# ##########
# ## Test ##
# ##########
enable_testing()
add_subdirectory(test)
