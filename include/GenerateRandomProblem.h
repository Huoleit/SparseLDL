#include "SparseLDL/Types.h"

namespace pdal {
template <typename Scalar>
inline tpl::CostApproximation<Scalar> getRandomCost(int n, int m) {
  matrix_s_t<Scalar> QPPR = matrix_s_t<Scalar>::Random(n + m, n + m);
  QPPR = QPPR.transpose() * QPPR;
  tpl::CostApproximation<Scalar> cost;
  cost.Q = QPPR.topLeftCorner(n, n);
  cost.R = QPPR.bottomRightCorner(m, m);
  return cost;
}

template <typename Scalar, int Nx, int Nu>
inline tpl::CostApproximation<Scalar, Nx, Nu> getRandomCost() {
  matrix_s_t<Scalar> QPPR = matrix_s_t<Scalar>::Random(Nx + Nu, Nx + Nu);
  QPPR = QPPR.transpose() * QPPR;
  tpl::CostApproximation<Scalar, Nx, Nu> cost;
  cost.Q = QPPR.template topLeftCorner<Nx, Nx>();
  cost.R = QPPR.template bottomRightCorner<Nu, Nu>();

  return cost;
}

template <typename Scalar>
inline tpl::DynamicsLinearApproximation<Scalar> getRandomDynamics(int n, int m) {
  tpl::DynamicsLinearApproximation<Scalar> dynamics;
  dynamics.A = matrix_s_t<Scalar>::Random(n, n);
  dynamics.B = matrix_s_t<Scalar>::Random(n, m);
  return dynamics;
}

template <typename Scalar, int Nx, int Nu>
inline tpl::DynamicsLinearApproximation<Scalar, Nx, Nu> getRandomDynamics() {
  tpl::DynamicsLinearApproximation<Scalar, Nx, Nu> dynamics;
  dynamics.A.setRandom();
  dynamics.B.setRandom();
  return dynamics;
}

template <typename Scalar>
inline tpl::CostApproximation<Scalar, 4, 2> getFixedCost() {
  tpl::CostApproximation<Scalar, 4, 2> cost;
  cost.setZero();

  cost.Q.diagonal() << 1, 1, 0.1, 0.1;
  cost.R.diagonal() << 0.3, 0.3;
  return cost;
}

template <typename Scalar>
inline tpl::DynamicsLinearApproximation<Scalar, 4, 2> getFixedDynamics() {
  tpl::DynamicsLinearApproximation<Scalar, 4, 2> dynamics;
  // A = [I(2) dt * I(2); zeros(2, 2) I(2)];
  // B = [0.5 * dt * dt * I(2); dt * I(2)];
  scalar_t dt = 0.1;
  dynamics.A << matrix_s_t<Scalar>::Identity(2, 2), dt * matrix_s_t<Scalar>::Identity(2, 2), matrix_s_t<Scalar>::Zero(2, 2),
      matrix_s_t<Scalar>::Identity(2, 2);

  dynamics.B << 0.5 * dt * dt * matrix_s_t<Scalar>::Identity(2, 2), dt * matrix_s_t<Scalar>::Identity(2, 2);
  return dynamics;
}

}  // namespace pdal
