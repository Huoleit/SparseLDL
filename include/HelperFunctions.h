#pragma once

#include <stdexcept>
#include <vector>

#include "SparseLDL/Types.h"

namespace pdal {

template <typename Scalar, int Nx, int Nu>
matrix_s_t<Scalar> getConstraintMatrix(const DynamicsAlignedStdVector<Scalar, Nx, Nu>& dynamics, int m, int n) {
  const int N = dynamics.size();
  if (N < 1) {
    throw std::runtime_error("[getConstraintMatrix] The number of stages cannot be less than 1.");
  }

  // Preallocate full constraint matrix
  matrix_s_t<Scalar> G(m, n);
  G.setZero();

  // Initial state constraint
  const int nu_0 = dynamics[0].B.cols();
  const int nx_1 = dynamics[0].B.rows();
  G.topLeftCorner(nx_1, nu_0 + nx_1) << dynamics.front().B, -matrix_s_t<Scalar>::Identity(nx_1, nx_1);

  int currRow = nx_1;
  int currCol = nu_0;
  for (int k = 1; k < N; ++k) {
    const auto& dynamics_k = dynamics[k];
    // const auto& constraints_k = constraints;
    const int nu_k = dynamics_k.B.cols();
    const int nx_k = dynamics_k.A.cols();
    const int nx_next = dynamics_k.A.rows();

    // Add [A, B, -I]
    G.block(currRow, currCol, nx_next, nx_k + nu_k + nx_next) << dynamics_k.A, dynamics_k.B,
        -matrix_s_t<Scalar>::Identity(nx_next, nx_next);

    currRow += nx_next;
    currCol += nx_k + nu_k;
  }

  return G;
}

template <typename Scalar, int Nx, int Nu>
matrix_s_t<Scalar> getCostMatrix(const CostAlignedStdVector<Scalar, Nx, Nu>& cost, const int m) {
  const int N = cost.size() - 1;

  // Preallocate full Cost matrices
  matrix_s_t<Scalar> H(m, m);
  H.setZero();

  const int nu_0 = cost[0].R.cols();
  H.topLeftCorner(nu_0, nu_0) = cost[0].R;

  int currRow = nu_0;
  for (int k = 1; k < N; ++k) {
    const int nx_k = cost[k].Q.cols();
    const int nu_k = cost[k].R.cols();

    // Add [ Q, 0
    //       0, R ]
    H.block(currRow, currRow, nx_k + nu_k, nx_k + nu_k) << cost[k].Q, matrix_s_t<Scalar>::Zero(nx_k, nu_k),
        matrix_s_t<Scalar>::Zero(nu_k, nx_k), cost[k].R;

    currRow += nx_k + nu_k;
  }

  const int nx_N = cost.back().Q.cols();
  H.bottomRightCorner(nx_N, nx_N) = cost[N].Q;

  return H;
}
}  // namespace pdal