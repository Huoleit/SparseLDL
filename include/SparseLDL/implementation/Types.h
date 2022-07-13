#pragma once

#include <Eigen/Core>

namespace pdal {
namespace tpl {
template <typename Scalar>
DynamicsLinearApproximation<Scalar>& DynamicsLinearApproximation<Scalar>::resize(size_t nx, size_t nu) {
  A.resize(nx, nx);
  B.resize(nx, nu);

  return *this;
}

template <typename Scalar>
DynamicsLinearApproximation<Scalar>& DynamicsLinearApproximation<Scalar>::setZero(size_t nx, size_t nu) {
  A.setZero(nx, nx);
  B.setZero(nx, nu);

  return *this;
}

template <typename Scalar>
DynamicsLinearApproximation<Scalar> DynamicsLinearApproximation<Scalar>::Zero(size_t nx, size_t nu) {
  DynamicsLinearApproximation<Scalar> f;
  f.setZero(nx, nu);

  return f;
}

template <typename Scalar>
CostApproximation<Scalar>& CostApproximation<Scalar>::resize(size_t nx, size_t nu) {
  Q.resize(nx, nx);
  R.resize(nu, nu);

  return *this;
}

template <typename Scalar>
CostApproximation<Scalar>& CostApproximation<Scalar>::setZero(size_t nx, size_t nu) {
  Q.setZero(nx, nx);
  R.setZero(nu, nu);

  return *this;
}

template <typename Scalar>
CostApproximation<Scalar> CostApproximation<Scalar>::Zero(size_t nx, size_t nu) {
  CostApproximation<Scalar> f;
  f.setZero(nx, nu);

  return f;
}
}  // namespace tpl
}  // namespace pdal