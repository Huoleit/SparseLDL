#include "SparseLDL/Types.h"

namespace pdal {
DynamicsLinearApproximation& DynamicsLinearApproximation::resize(size_t nx, size_t nu) {
  A.resize(nx, nx);
  B.resize(nx, nu);

  return *this;
}

DynamicsLinearApproximation& DynamicsLinearApproximation::setZero(size_t nx, size_t nu) {
  A.setZero(nx, nx);
  B.setZero(nx, nu);

  return *this;
}

DynamicsLinearApproximation DynamicsLinearApproximation::Zero(size_t nx, size_t nu) {
  DynamicsLinearApproximation f;
  f.setZero(nx, nu);

  return f;
}

CostApproximation& CostApproximation::resize(size_t nx, size_t nu) {
  Q.resize(nx, nx);
  R.resize(nx, nu);

  return *this;
}

CostApproximation& CostApproximation::setZero(size_t nx, size_t nu) {
  Q.setZero(nx, nx);
  R.setZero(nx, nu);

  return *this;
}

CostApproximation CostApproximation::Zero(size_t nx, size_t nu) {
  CostApproximation f;
  f.setZero(nx, nu);

  return f;
}

}  // namespace pdal
