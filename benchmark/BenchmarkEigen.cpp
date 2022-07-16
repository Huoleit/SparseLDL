#include <vector>

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <Eigen/StdVector>

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

template <typename SCALAR, int ROW, int COL = ROW>
void BM_MatrixMulFixedSize(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, ROW, COL> A;
  Matrix<SCALAR, COL, COL> B;
  A.setRandom();
  B.setRandom();

  for (auto _ : state) {
    benchmark::DoNotOptimize((A * B).eval());
  }
}

template <typename SCALAR>
void BM_MatrixInverse(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, -1, -1> A, B;
  A.setRandom(state.range(0), state.range(0));
  B.setRandom(state.range(0), state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(A.template selfadjointView<Eigen::Upper>().ldlt().solve(B).eval());
  }
}

template <typename SCALAR, int SIZE>
void BM_FixedMatrixWithStdVector(benchmark::State& state) {
  srand(0);
  std::vector<Matrix<SCALAR, SIZE, SIZE>, aligned_allocator<Matrix<SCALAR, SIZE, SIZE>>> v(3);
  v[0].setRandom();
  v[1].setRandom();

  for (auto _ : state) {
    v[2].noalias() = v[0] * v[1];
    benchmark::DoNotOptimize(v[2]);
  }
}

template <typename SCALAR, int SIZE>
void BM_MatrixCholInverse(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, SIZE, SIZE> A, B;
  A.setRandom();
  Matrix<SCALAR, SIZE, SIZE> I;
  I.setIdentity();
  A += I;

  B.setRandom();

  for (auto _ : state) {
    benchmark::DoNotOptimize(A.template selfadjointView<Eigen::Upper>().ldlt().solve(B).eval());
  }
}

template <typename SCALAR, int SIZE>
void BM_MatrixLUInverse(benchmark::State& state) {
  srand(0);
  Matrix<SCALAR, SIZE, SIZE> A, B;
  A = Matrix<SCALAR, SIZE, SIZE>::Random().template selfadjointView<Eigen::Upper>();
  Matrix<SCALAR, SIZE, SIZE> I;
  I.setIdentity();
  A += I;

  for (auto _ : state) {
    benchmark::DoNotOptimize((A.inverse() * B).eval());
  }
}

// BENCHMARK_TEMPLATE(BM_MatrixMulDynamicsSize, float)->RangeMultiplier(2)->Range(2, 2 << 10);
// BENCHMARK_TEMPLATE(BM_MatrixMulDynamicsSize, double)->RangeMultiplier(2)->Range(2, 2 << 10);

// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 2);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, float, 2);
// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 2);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, double, 2);

// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 4);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, float, 4);
// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 4);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, double, 4);

// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 6);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, float, 6);
// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 6);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, double, 6);

// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 8);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, float, 8);
// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 8);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, double, 8);

// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, float, 10);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, float, 10);
// BENCHMARK_TEMPLATE2(BM_MatrixMulFixedSize, double, 10);
// BENCHMARK_TEMPLATE2(BM_FixedMatrixWithStdVector, double, 10);

// BENCHMARK_TEMPLATE(BM_MatrixInverse, float)->DenseRange(2, 10);
// BENCHMARK_TEMPLATE(BM_MatrixInverse, double)->DenseRange(2, 10);

BENCHMARK(BM_MatrixCholInverse<float, 4>);
BENCHMARK(BM_MatrixCholInverse<double, 4>);
BENCHMARK(BM_MatrixLUInverse<float, 4>);
BENCHMARK(BM_MatrixLUInverse<double, 4>);

BENCHMARK(BM_MatrixCholInverse<float, 8>);
BENCHMARK(BM_MatrixCholInverse<double, 8>);
BENCHMARK(BM_MatrixLUInverse<float, 8>);
BENCHMARK(BM_MatrixLUInverse<double, 8>);

BENCHMARK(BM_MatrixCholInverse<float, 16>);
BENCHMARK(BM_MatrixCholInverse<double, 16>);
BENCHMARK(BM_MatrixLUInverse<float, 16>);
BENCHMARK(BM_MatrixLUInverse<double, 16>);
