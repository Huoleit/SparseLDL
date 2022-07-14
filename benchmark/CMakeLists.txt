set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz
)
FetchContent_MakeAvailable(googlebenchmark)

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
target_compile_options(BenchmarkEigen PRIVATE -Wfatal-errors -march=native -msse2 -mavx2 -mavx)

add_executable(BenchmarkSparseLDL
  BenchmarkSparseLDL.cpp
)
target_link_libraries(BenchmarkSparseLDL
  ${PROJECT_NAME}
  benchmark::benchmark_main
)