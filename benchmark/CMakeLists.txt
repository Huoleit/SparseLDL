set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz
)
FetchContent_MakeAvailable(googlebenchmark)

set(QDLDL_BUILD_SHARED_LIB OFF)
set(QDLDL_BUILD_DEMO_EXE OFF)
set(QDLDL_UNITTESTS OFF)
set(QDLDL_LONG OFF)
FetchContent_Declare(
  qdldl
  URL https://github.com/osqp/qdldl/archive/refs/tags/v0.1.6.tar.gz
)
FetchContent_MakeAvailable(qdldl)

# #############
# ## Targets ##
# #############
add_executable(BenchmarkEigen
  BenchmarkEigen.cpp
)
target_link_libraries(BenchmarkEigen
  Eigen3::Eigen
  benchmark::benchmark_main
)

# #####
add_executable(BenchmarkSparseLDL
  BenchmarkSparseLDL.cpp
)
target_link_libraries(BenchmarkSparseLDL
  ${PROJECT_NAME}
  qdldlstatic
  benchmark::benchmark_main
)

# #####
add_executable(SimpleEigenBenchmark
  SimpleEigenBenchmark.cpp
)
target_link_libraries(SimpleEigenBenchmark
  Eigen3::Eigen
)
target_compile_options(SimpleEigenBenchmark PRIVATE
  -save-temps
)