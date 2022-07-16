#include "SparseLDL/CodeGen/SparseLDLGen.h"
#include "SparseLDL/SparseLDL.h"

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;

template <typename Scalar>
class BenchmarkSparseLDL : public benchmark::Fixture {
 public:
  constexpr static const size_t N = 2;  // numStages
  constexpr static const size_t nx = 12;
  constexpr static const size_t nu = 12;
  constexpr static const size_t numDecisionVariables = N * (nx + nu);
  constexpr static const size_t numConstraints = N * nx;

  BenchmarkSparseLDL() {
    srand(0);

    for (int i = 0; i < N; i++) {
      dynamics.push_back(getRandomDynamics<Scalar, nx, nu>());
      cost.push_back(getRandomCost<Scalar, nx, nu>());
    }
    cost.push_back(getRandomCost<Scalar, nx, nu>());

    b = vector_s_t<Scalar>::Ones(numDecisionVariables + numConstraints);

    H = getCostMatrix(cost, numDecisionVariables);
    G = getConstraintMatrix(dynamics, numConstraints, numDecisionVariables);
    KKT.resize(numConstraints + numDecisionVariables, numConstraints + numDecisionVariables);
    KKT << H, G.transpose(), G, -eps * matrix_s_t<Scalar>::Identity(numConstraints, numConstraints);
  }

  DynamicsAlignedStdVector<Scalar, nx, nu> dynamics;
  CostAlignedStdVector<Scalar, nx, nu> cost;
  Scalar eps{1e-5};

  std::vector<matrix_s_t<Scalar>> DInv;
  std::vector<matrix_s_t<Scalar>> Lx;
  std::vector<matrix_s_t<Scalar>> D;

  DxCollection<Scalar, nx, nu> DxCollection_;
  DxCollection<Scalar, nx, nu> DxInvCollection_;
  LxCollection<Scalar, nx, nu> LxCollection_;

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
    solve(Lx, DInv, tmp);
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, SparseLDLDouble, double)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    sparseLDL(dynamics, cost, Lx, DInv, D, eps);
    solve(Lx, DInv, tmp);
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, SparseLDL2StageFloat, float)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    solveWithSparseLDL(dynamics, cost, LxCollection_, DxCollection_, DxInvCollection_, tmp);
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, SparseLDL2StageDouble, double)(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    solveWithSparseLDL(dynamics, cost, LxCollection_, DxCollection_, DxInvCollection_, tmp);
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

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, EigenLUFloat, float)(benchmark::State& state) {
  for (auto _ : state) {
    KKT.lu().solve(b).eval();
  }
}

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, EigenLUDouble, double)(benchmark::State& state) {
  for (auto _ : state) {
    KKT.lu().solve(b).eval();
  }
}
