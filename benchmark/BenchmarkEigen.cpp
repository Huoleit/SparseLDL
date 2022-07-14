#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>

using namespace Eigen;

template <typename SCALAR>
void BM_MatrixMulDynamicsSize(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, Dynamic, Dynamic> A(state.range(0), state.range(0)), B(state.range(0), state.range(0));
  A.setRandom();
  B.setRandom();

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
  }
}

template <typename SCALAR, int SIZE>
void BM_MatrixMulFixedSize(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, SIZE, SIZE> A, B;
  A.setRandom();
  B.setRandom();

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
  }
}

template <typename SCALAR, int SIZE>
void BM_MatrixInverse(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, SIZE, SIZE> A, B;
  A.setRandom();
  B.setRandom();

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
  }
}

// BENCHMARK_TEMPLATE(BM_MatrixMulDynamicsSize, float)->RangeMultiplier(2)->Range(2, 2 << 10);
// BENCHMARK_TEMPLATE(BM_MatrixMulDynamicsSize, double)->RangeMultiplier(2)->Range(2, 2 << 10);

BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 2);
BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 2);

BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 4);
BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 4);

BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 6);
BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 6);

BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 8);
BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 8);

BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 10);
BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 10);
