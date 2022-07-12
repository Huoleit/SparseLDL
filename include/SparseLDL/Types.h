#pragma once

#include <Eigen/Core>

#include <ostream>

namespace pdal {

using scalar_t = double;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

struct DynamicsLinearApproximation {
  matrix_t A;
  matrix_t B;

  DynamicsLinearApproximation& resize(size_t nx, size_t nu);
  DynamicsLinearApproximation& setZero(size_t nx, size_t nu);
  static DynamicsLinearApproximation Zero(size_t nx, size_t nu);
};

struct CostApproximation {
  // Assume diagonal cost
  matrix_t Q;
  matrix_t R;

  CostApproximation& resize(size_t nx, size_t nu);
  CostApproximation& setZero(size_t nx, size_t nu);
  static CostApproximation Zero(size_t nx, size_t nu);
};
}  // namespace pdal