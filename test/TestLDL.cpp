#include "SparseLDL/CodeGen/SparseLDLGen.h"
#include "SparseLDL/SparseLDL.h"

#include <iostream>

#include <gtest/gtest.h>
#include <Eigen/Cholesky>

#include "GenerateRandomProblem.h"
#include "HelperFunctions.h"
#include "SparseLDL/Types.h"

using namespace pdal;
using namespace std;

Eigen::VectorXi indexFromLength(int from, int length) {
  Eigen::VectorXi ind(length);
  for (int i = 0; i < length; ++i) ind(i) = from + i;
  return ind;
}
class TestLDL : public ::testing::Test {
 public:
  constexpr static const int N = 4;  // numStages
  constexpr static const int nx = 8;
  constexpr static const int nu = 4;
  constexpr static const int numDecisionVariables = N * (nx + nu);
  constexpr static const int numConstraints = N * nx;

  TestLDL() {
    srand(0);

    for (int i = 0; i < N; ++i) {
      perm.indices().segment((nu + nx + nx) * i, nu) = indexFromLength((nu + nx) * i, nu);
      perm.indices().segment((nu + nx + nx) * i + nu, nx) = indexFromLength(numDecisionVariables + i * nx, nx);
      perm.indices().segment((nu + nx + nx) * i + nu + nx, nx) = indexFromLength((nu + nx) * i + nu, nx);
    }

    for (int i = 0; i < N; i++) {
      dynamics.push_back(getRandomDynamics<double>(nx, nu));
      cost.push_back(getRandomCost<double>(nx, nu));
    }
    cost.push_back(getRandomCost<double>(nx, nu));
  }

  Eigen::PermutationMatrix<numDecisionVariables + numConstraints, numDecisionVariables + numConstraints> perm;
  DynamicsAlignedStdVector<double, -1, -1> dynamics;
  CostAlignedStdVector<double, -1, -1> cost;
  double eps{1e-5};
};

TEST_F(TestLDL, Decomposition) {
  std::vector<matrix_t> DInv;
  std::vector<matrix_t> Lx;
  std::vector<matrix_t> D;

  matrix_t H = getCostMatrix(cost, numDecisionVariables);
  matrix_t G = getConstraintMatrix(dynamics, numConstraints, numDecisionVariables);
  matrix_t KKT(numConstraints + numDecisionVariables, numConstraints + numDecisionVariables);
  KKT << H, G.transpose(), G, -eps * matrix_t::Identity(numConstraints, numConstraints);

  vector_t b = vector_t::Ones(numDecisionVariables + numConstraints);
  vector_t reference = perm.transpose() * KKT.ldlt().solve(b);

  sparseLDL(dynamics, cost, Lx, DInv, D, eps);
  // Solve in-place
  solve(Lx, DInv, b);

  EXPECT_TRUE(reference.isApprox(b, 1e-6)) << "|ref - b|_inf = " << (reference - b).lpNorm<Eigen::Infinity>();
}

TEST(LDL, CodeGen) {
  srand(10);
  constexpr static const size_t N = 2;  // numStages
  constexpr static const size_t nx = 4;
  constexpr static const size_t nu = 2;
  constexpr static const size_t numDecisionVariables = N * (nx + nu);
  constexpr static const size_t numConstraints = N * nx;

  DynamicsAlignedStdVector<double, nx, nu> dynamics;
  CostAlignedStdVector<double, nx, nu> cost;

  for (int i = 0; i < N; i++) {
    dynamics.push_back(getRandomDynamics<double, nx, nu>());
    cost.push_back(getRandomCost<double, nx, nu>());
  }
  cost.push_back(getRandomCost<double, nx, nu>());

  vector_t b = vector_t::Ones(numDecisionVariables + numConstraints);
  {
    DxCollection<double, nx, nu> Dx;
    DxCollection<double, nx, nu> DxInv;
    LxCollection<double, nx, nu> Lx;
    solveWithSparseLDL(dynamics, cost, Lx, Dx, DxInv, b);
  }

  vector_t b_ref = vector_t::Ones(numDecisionVariables + numConstraints);
  {
    std::vector<matrix_t> DxInv;
    std::vector<matrix_t> Lx;
    std::vector<matrix_t> Dx;

    sparseLDL(dynamics, cost, Lx, DxInv, Dx);
    solve(Lx, DxInv, b_ref);
  }

  EXPECT_TRUE(b.isApprox(b_ref, 1e-4)) << "|ref - b|_inf = " << (b_ref - b).lpNorm<Eigen::Infinity>();
}