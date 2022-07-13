#pragma once

#include <Eigen/Core>

namespace pdal {

using scalar_t = double;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using matrix_s_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <typename Scalar>
using vector_s_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

namespace tpl {
template <typename Scalar>
struct DynamicsLinearApproximation {
  matrix_s_t<Scalar> A;
  matrix_s_t<Scalar> B;

  template <typename Other>
  DynamicsLinearApproximation<Other> cast() {
    DynamicsLinearApproximation<Other> res;
    res.A = A.template cast<Other>();
    res.B = B.template cast<Other>();

    return res;
  };
  DynamicsLinearApproximation& resize(size_t nx, size_t nu);
  DynamicsLinearApproximation& setZero(size_t nx, size_t nu);
  static DynamicsLinearApproximation Zero(size_t nx, size_t nu);
};

template <typename Scalar>
struct CostApproximation {
  // Assume diagonal cost
  matrix_s_t<Scalar> Q;
  matrix_s_t<Scalar> R;

  template <typename Other>
  CostApproximation<Other> cast() {
    CostApproximation<Other> res;
    res.Q = Q.template cast<Other>();
    res.R = R.template cast<Other>();

    return res;
  };
  CostApproximation& resize(size_t nx, size_t nu);
  CostApproximation& setZero(size_t nx, size_t nu);
  static CostApproximation Zero(size_t nx, size_t nu);
};
}  // namespace tpl

using DynamicsLinearApproximationFloat = tpl::DynamicsLinearApproximation<float>;
using DynamicsLinearApproximationDouble = tpl::DynamicsLinearApproximation<double>;
using CostApproximationFloat = tpl::CostApproximation<float>;
using CostApproximationDouble = tpl::CostApproximation<double>;

}  // namespace pdal

extern template struct pdal::tpl::DynamicsLinearApproximation<float>;
extern template struct pdal::tpl::DynamicsLinearApproximation<double>;
extern template struct pdal::tpl::CostApproximation<float>;
extern template struct pdal::tpl::CostApproximation<double>;