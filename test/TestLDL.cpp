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
  constexpr static const size_t N = 5;  // numStages
  constexpr static const size_t nx = 8;
  constexpr static const size_t nu = 8;
  constexpr static const size_t numDecisionVariables = N * (nx + nu);
  constexpr static const size_t numConstraints = N * nx;

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
  std::vector<DynamicsLinearApproximationDouble> dynamics;
  std::vector<CostApproximationDouble> cost;
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