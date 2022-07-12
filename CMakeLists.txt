cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(SparseLDL VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

add_library(SparseLDL
  src/Types.cpp
  src/SparseLDL.cpp
)
target_link_libraries(SparseLDL
  Eigen3::Eigen
)

# ###############
# ## Benchmark ##
# ###############
add_subdirectory(benchmark)

# ##########
# ## Test ##
# ##########
enable_testing()
include(GoogleTest)
find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}-test
  test/TestLDL.cpp
  test/GenerateRandomProblem.h
)
target_link_libraries(${PROJECT_NAME}-test
  SparseLDL
  Eigen3::Eigen
  GTest::Main
)
gtest_discover_tests(${PROJECT_NAME}-test)