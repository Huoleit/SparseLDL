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

template <typename Scalar>
inline tpl::CostApproximation<Scalar> getRandomDiagonalCost(int n, int m) {
  matrix_s_t<Scalar> QPPR = matrix_s_t<Scalar>::Random(n + m, n + m);
  QPPR = QPPR.transpose() * QPPR;
  tpl::CostApproximation<Scalar> cost;
  cost.Q = QPPR.topLeftCorner(n, n).diagonal().asDiagonal();
  cost.R = QPPR.bottomRightCorner(m, m).diagonal().asDiagonal();
  return cost;
}

template <typename Scalar>
inline tpl::DynamicsLinearApproximation<Scalar> getRandomDynamics(int n, int m) {
  tpl::DynamicsLinearApproximation<Scalar> dynamics;
  dynamics.A = matrix_s_t<Scalar>::Random(n, n);
  dynamics.B = matrix_s_t<Scalar>::Random(n, m);
  return dynamics;
}

template <typename Scalar>
inline tpl::CostApproximation<Scalar> getFixedCost() {
  tpl::CostApproximation<Scalar> cost;
  cost.setZero(4, 2);

  cost.Q.diagonal() << 1, 1, 0.1, 0.1;
  cost.R.diagonal() << 0.3, 0.3;
  return cost;
}

template <typename Scalar>
inline tpl::DynamicsLinearApproximation<Scalar> getFixedDynamics() {
  tpl::DynamicsLinearApproximation<Scalar> dynamics;
  // A = [I(2) dt * I(2); zeros(2, 2) I(2)];
  // B = [0.5 * dt * dt * I(2); dt * I(2)];
  scalar_t dt = 0.1;
  dynamics.A = (matrix_s_t<Scalar>(4, 4) << matrix_s_t<Scalar>::Identity(2, 2), dt * matrix_s_t<Scalar>::Identity(2, 2),
                matrix_s_t<Scalar>::Zero(2, 2), matrix_s_t<Scalar>::Identity(2, 2))
                   .finished();
  dynamics.B =
      (matrix_s_t<Scalar>(4, 2) << 0.5 * dt * dt * matrix_s_t<Scalar>::Identity(2, 2), dt * matrix_s_t<Scalar>::Identity(2, 2)).finished();
  return dynamics;
}

}  // namespace pdal
