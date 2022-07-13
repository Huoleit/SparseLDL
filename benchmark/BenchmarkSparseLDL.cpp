#include "SparseLDL/SparseLDL.h"

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;

class BenchmarkSparseLDL : public benchmark::Fixture {
 public:
  constexpr static const size_t N = 2;  // numStages
  constexpr static const size_t nx = 4;
  constexpr static const size_t nu = 2;
  constexpr static const size_t numDecisionVariables = N * (nx + nu);
  constexpr static const size_t numConstraints = N * nx;

  BenchmarkSparseLDL() {
    srand(0);

    for (int i = 0; i < N; i++) {
      dynamicsDouble.push_back(getFixedDynamics<double>());
      costDouble.push_back(getFixedCost<double>());

      dynamicsFloat.push_back(dynamicsDouble.back().cast<float>());
      costFloat.push_back(costDouble.back().cast<float>());
    }
    costDouble.push_back(getFixedCost<double>());
    costFloat.push_back(costDouble.back().cast<float>());
  }

  std::vector<DynamicsLinearApproximationDouble> dynamicsDouble;
  std::vector<CostApproximationDouble> costDouble;
  std::vector<DynamicsLinearApproximationFloat> dynamicsFloat;
  std::vector<CostApproximationFloat> costFloat;
  double eps{1e-5};
};

BENCHMARK_F(BenchmarkSparseLDL, EigenCholeskyDouble)(benchmark::State& state) {
  matrix_t H = getCostMatrix(costDouble, numDecisionVariables);
  matrix_t G = getConstraintMatrix(dynamicsDouble, numConstraints, numDecisionVariables);
  matrix_t KKT(numConstraints + numDecisionVariables, numConstraints + numDecisionVariables);
  KKT << H, G.transpose(), G, -eps * matrix_t::Identity(numConstraints, numConstraints);

  vector_t b = vector_t::Ones(numDecisionVariables + numConstraints);

  for (auto _ : state) {
    KKT.ldlt().solve(b).eval();
  }
}

BENCHMARK_F(BenchmarkSparseLDL, EigenCholeskyFloat)(benchmark::State& state) {
  matrix_s_t<float> H = getCostMatrix(costFloat, numDecisionVariables);
  matrix_s_t<float> G = getConstraintMatrix(dynamicsFloat, numConstraints, numDecisionVariables);
  matrix_s_t<float> KKT(numConstraints + numDecisionVariables, numConstraints + numDecisionVariables);
  KKT << H, G.transpose(), G, -eps * matrix_s_t<float>::Identity(numConstraints, numConstraints);

  vector_s_t<float> b = vector_s_t<float>::Ones(numDecisionVariables + numConstraints);

  for (auto _ : state) {
    KKT.ldlt().solve(b).eval();
  }
}

BENCHMARK_F(BenchmarkSparseLDL, SparseLDLDouble)(benchmark::State& state) {
  std::vector<matrix_t> DInv;
  std::vector<matrix_t> Lx;
  std::vector<matrix_t> D;
  vector_t b = vector_t::Ones(numDecisionVariables + numConstraints);
  vector_t tmp = b;

  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    sparseLDL(dynamicsDouble, costDouble, Lx, DInv, D, eps);
    solve(Lx, DInv, tmp);
  }
}

BENCHMARK_F(BenchmarkSparseLDL, SparseLDLFloat)(benchmark::State& state) {
  std::vector<matrix_s_t<float>> DInv;
  std::vector<matrix_s_t<float>> Lx;
  std::vector<matrix_s_t<float>> D;
  vector_s_t<float> b = vector_s_t<float>::Ones(numDecisionVariables + numConstraints);
  vector_s_t<float> tmp = b;

  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    sparseLDL(dynamicsFloat, costFloat, Lx, DInv, D, static_cast<float>(eps));
    solve(Lx, DInv, b);
  }
}
