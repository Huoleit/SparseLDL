#include "SparseLDL/SparseLDL.h"

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;

template <typename Scalar>
class BenchmarkSparseLDL : public benchmark::Fixture {
 public:
  constexpr static const size_t N = 4;  // numStages
  constexpr static const size_t nx = 8;
  constexpr static const size_t nu = 8;
  constexpr static const size_t numDecisionVariables = N * (nx + nu);
  constexpr static const size_t numConstraints = N * nx;

  BenchmarkSparseLDL() {
    srand(0);

    for (int i = 0; i < N; i++) {
      dynamics.push_back(getRandomDynamics<Scalar>(nx, nu));
      cost.push_back(getRandomCost<Scalar>(nx, nu));
    }
    cost.push_back(getRandomCost<Scalar>(nx, nu));

    b = vector_s_t<Scalar>::Ones(numDecisionVariables + numConstraints);

    H = getCostMatrix(cost, numDecisionVariables);
    G = getConstraintMatrix(dynamics, numConstraints, numDecisionVariables);
    KKT.resize(numConstraints + numDecisionVariables, numConstraints + numDecisionVariables);
    KKT << H, G.transpose(), G, -eps * matrix_s_t<Scalar>::Identity(numConstraints, numConstraints);
  }

  std::vector<tpl::DynamicsLinearApproximation<Scalar>> dynamics;
  std::vector<tpl::CostApproximation<Scalar>> cost;
  Scalar eps{1e-5};

  std::vector<matrix_s_t<Scalar>> DInv;
  std::vector<matrix_s_t<Scalar>> Lx;
  std::vector<matrix_s_t<Scalar>> D;

  vector_s_t<Scalar> b;
  vector_s_t<Scalar> tmp;

  matrix_s_t<Scalar> H;
  matrix_s_t<Scalar> G;
  matrix_s_t<Scalar> KKT;
};

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, SparseLDLFloat, float)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    sparseLDL(dynamics, cost, Lx, DInv, D, eps);
    solve(Lx, DInv, b);
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, SparseLDLDouble, double)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    sparseLDL(dynamics, cost, Lx, DInv, D, eps);
    solve(Lx, DInv, b);
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, EigenCholeskyFloat, float)(benchmark::State& state) {
  for (auto _ : state) {
    KKT.ldlt().solve(b).eval();
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, EigenCholeskyDouble, double)(benchmark::State& state) {
  for (auto _ : state) {
    KKT.ldlt().solve(b).eval();
  }
}
