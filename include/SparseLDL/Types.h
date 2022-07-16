#pragma once

#include <type_traits>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace pdal {

using scalar_t = double;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using matrix_s_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <typename Scalar>
using vector_s_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

namespace tpl {
template <typename Scalar, int Nx = -1, int Nu = -1>
struct DynamicsLinearApproximation {
  static_assert((Nx != -1 && Nu != -1) || (Nx == -1 && Nu == -1), "Nx and Nu must be either -1 or both non-negative");
  Eigen::Matrix<Scalar, Nx, Nx> A;
  Eigen::Matrix<Scalar, Nx, Nu> B;

  // Fixed-size version
  template <int SIZE = Nx, typename std::enable_if<SIZE != -1>::type>
  DynamicsLinearApproximation& setZero() {
    A.setZero();
    B.setZero();
    return *this;
  }

  // Dynamic-size version
  template <int SIZE = Nx, typename std::enable_if<SIZE == -1>::type>
  DynamicsLinearApproximation& setZero(size_t nx, size_t nu) {
    A.setZero(nx, nx);
    B.setZero(nx, nu);
    return *this;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename Scalar, int Nx = -1, int Nu = -1>
struct CostApproximation {
  static_assert((Nx != -1 && Nu != -1) || (Nx == -1 && Nu == -1), "Nx and Nu must be either -1 or both non-negative");

  Eigen::Matrix<Scalar, Nx, Nx> Q;
  Eigen::Matrix<Scalar, Nu, Nu> R;

  // Fixed-size version
  template <int SIZE = Nx, typename std::enable_if<SIZE != -1>::type>
  CostApproximation& setZero() {
    Q.setZero();
    R.setZero();
    return *this;
  }

  // Dynamic-size version
  template <int SIZE = Nx, typename std::enable_if<SIZE == -1>::type>
  CostApproximation& setZero(size_t nx, size_t nu) {
    Q.setZero(nx, nx);
    R.setZero(nu, nu);
    return *this;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace tpl

template <int Nx, int Nu>
using DynamicsApproximationFloat = tpl::DynamicsLinearApproximation<float>;
using DynamicsApproximationDouble = tpl::DynamicsLinearApproximation<double>;
using CostApproximationFloat = tpl::CostApproximation<float>;
using CostApproximationDouble = tpl::CostApproximation<double>;

template <typename Scalar, int Nx, int Nu>
using DynamicsAlignedStdVector = std::vector<tpl::DynamicsLinearApproximation<Scalar, Nx, Nu>,
                                             Eigen::aligned_allocator<tpl::DynamicsLinearApproximation<Scalar, Nx, Nu>>>;

template <typename Scalar, int Nx, int Nu>
using CostAlignedStdVector =
    std::vector<tpl::CostApproximation<Scalar, Nx, Nu>, Eigen::aligned_allocator<tpl::CostApproximation<Scalar, Nx, Nu>>>;

}  // namespace pdal

extern template struct pdal::tpl::DynamicsLinearApproximation<float>;
extern template struct pdal::tpl::DynamicsLinearApproximation<double>;
extern template struct pdal::tpl::CostApproximation<float>;
extern template struct pdal::tpl::CostApproximation<double>;