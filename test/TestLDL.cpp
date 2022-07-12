#include "SparseLDL/SparseLDL.h"

#include "GenerateRandomProblem.h"
#include "SparseLDL/Types.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace pdal;

constexpr size_t N_ = 2;  // numStages
constexpr size_t nx_ = 4;
constexpr size_t nu_ = 2;
constexpr size_t numDecisionVariables = N_ * (nx_ + nu_);
constexpr size_t numConstraints = N_ * nx_;

TEST(LDL, decomposition) {
  srand(0);
  std::vector<DynamicsLinearApproximation> dynamics;
  std::vector<CostApproximation> cost;

  for (int i = 0; i < N_; i++) {
    dynamics.push_back(getFixedDynamics());
    cost.push_back(getFixedCost());
  }
  cost.push_back(getFixedCost());

  std::vector<matrix_t> DInv;
  std::vector<matrix_t> Lx;
  std::vector<matrix_t> D;

  sparseLDL(dynamics, cost, Lx, DInv, D);

  vector_t b = vector_t::Ones(numDecisionVariables + numConstraints);

  solve(Lx, DInv, b);

  std::cerr << "Lx:"
            << "\n";
  for (const auto& m : Lx) {
    std::cerr << m << "\n\n";
  }

  std::cerr << "D:"
            << "\n";
  for (const auto& m : D) {
    std::cerr << m << "\n\n";
  }

  std::cerr << "DInv:"
            << "\n";
  for (const auto& m : DInv) {
    std::cerr << m << "\n\n";
  }

  std::cerr << "b: " << b.transpose() << "\n\n";
}