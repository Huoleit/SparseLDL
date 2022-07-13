#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>

using namespace Eigen;

static void BM_FixedSizeLDLTFloat(benchmark::State& state) {
  srand(0);
  Matrix<float, 16, 16> fixedMatrix;
  Matrix<float, 16, 1> fixedVector;
  fixedMatrix.setRandom();
  fixedVector.setRandom();
  fixedMatrix = fixedMatrix * fixedMatrix.transpose();

  for (auto _ : state) {
    benchmark::DoNotOptimize(fixedMatrix.ldlt().solve(fixedVector));
  }
}

static void BM_DynamicSizeLDLTFloat(benchmark::State& state) {
  srand(0);
  MatrixXf dynamicSizeMatrix(16, 16);
  VectorXf dynamicSizeVector(16);
  dynamicSizeMatrix.setRandom();
  dynamicSizeVector.setRandom();
  dynamicSizeMatrix = dynamicSizeMatrix * dynamicSizeMatrix.transpose();

  for (auto _ : state) {
    benchmark::DoNotOptimize(dynamicSizeMatrix.ldlt().solve(dynamicSizeVector));
  }
}

static void BM_FixedSizeMulFloat(benchmark::State& state) {
  srand(0);
  Matrix<double, 16, 16> A, B;
  A.setRandom();
  B.setRandom();
  benchmark::DoNotOptimize(A.data());
  benchmark::DoNotOptimize(B.data());

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
    benchmark::ClobberMemory();
  }
}

static void BM_DynamicSizeMulFloat(benchmark::State& state) {
  srand(0);
  MatrixXf A(16, 16), B(16, 16);
  A.setRandom();
  B.setRandom();
  benchmark::DoNotOptimize(A.data());
  benchmark::DoNotOptimize(B.data());

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
    benchmark::ClobberMemory();
  }
}
static void BM_DynamicSizeMulDouble(benchmark::State& state) {
  srand(0);
  MatrixXd A(16, 16), B(16, 16);
  A.setRandom();
  B.setRandom();
  benchmark::DoNotOptimize(A.data());
  benchmark::DoNotOptimize(B.data());

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_FixedSizeLDLTFloat);
BENCHMARK(BM_DynamicSizeLDLTFloat);
BENCHMARK(BM_FixedSizeMulFloat);
BENCHMARK(BM_DynamicSizeMulFloat);
BENCHMARK(BM_DynamicSizeMulDouble);
