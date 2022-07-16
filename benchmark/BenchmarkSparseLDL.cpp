#include "SparseLDL/CodeGen/SparseLDLGen.h"
#include "SparseLDL/SparseLDL.h"

#include <iostream>

#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SparseCore>

#include <qdldl.h>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;
using namespace Eigen;
using namespace std;

template <typename Scalar>
class BenchmarkSparseLDL : public benchmark::Fixture {
 public:
  constexpr static const size_t N = 2;  // numStages
  constexpr static const size_t nx = 16;
  constexpr static const size_t nu = 16;
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

  vector<matrix_s_t<Scalar>> DInv;
  vector<matrix_s_t<Scalar>> Lx;
  vector<matrix_s_t<Scalar>> D;

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

BENCHMARK_TEMPLATE_F(BenchmarkSparseLDL, QDLDL, QDLDL_float)(benchmark::State& state) {
  matrix_s_t<QDLDL_float> mTmp = KKT.triangularView<Upper>();
  SparseMatrix<QDLDL_float> hessian = mTmp.sparseView();

  constexpr QDLDL_int Hn = numDecisionVariables + numConstraints;
  vector<QDLDL_int> Lnz(Hn), etree(Hn);
  vector<QDLDL_int> flag(Hn);
  QDLDL_int sumLnz = QDLDL_etree(Hn, hessian.outerIndexPtr(), hessian.innerIndexPtr(), flag.data(), Lnz.data(), etree.data());

  if (sumLnz == -1) {
    cout << "QDLDL_etree failed" << endl;
    exit(1);
  }

  vector<QDLDL_int> Lp(Hn + 1), Li(sumLnz);
  vector<QDLDL_float> Lx(sumLnz);
  vector<QDLDL_float> D(Hn), Dinv(Hn);
  vector<QDLDL_bool> bwork(Hn);
  vector<QDLDL_int> iwork(3 * Hn);
  vector<QDLDL_float> fwork(Hn);

  for (auto _ : state) {
    state.PauseTiming();
    tmp = b;
    state.ResumeTiming();

    QDLDL_factor(Hn, hessian.outerIndexPtr(), hessian.innerIndexPtr(), hessian.valuePtr(), Lp.data(), Li.data(), Lx.data(), D.data(),
                 Dinv.data(), Lnz.data(), etree.data(), bwork.data(), iwork.data(), fwork.data());
    QDLDL_solve(Hn, Lp.data(), Li.data(), Lx.data(), Dinv.data(), tmp.data());
  }
}