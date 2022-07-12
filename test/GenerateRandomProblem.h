#include "SparseLDL/Types.h"

namespace pdal {
inline CostApproximation getRandomCost(int n, int m) {
  matrix_t QPPR = matrix_t::Random(n + m, n + m);
  QPPR = QPPR.transpose() * QPPR;
  CostApproximation cost;
  cost.Q = QPPR.topLeftCorner(n, n);
  cost.R = QPPR.bottomRightCorner(m, m);
  return cost;
}

inline CostApproximation getRandomDiagonalCost(int n, int m) {
  matrix_t QPPR = matrix_t::Random(n + m, n + m);
  QPPR = QPPR.transpose() * QPPR;
  CostApproximation cost;
  cost.Q = QPPR.topLeftCorner(n, n).diagonal().asDiagonal();
  cost.R = QPPR.bottomRightCorner(m, m).diagonal().asDiagonal();
  return cost;
}

inline DynamicsLinearApproximation getRandomDynamics(int n, int m) {
  DynamicsLinearApproximation dynamics;
  dynamics.A = matrix_t::Random(n, n);
  dynamics.B = matrix_t::Random(n, m);
  return dynamics;
}

inline CostApproximation getFixedCost() {
  CostApproximation cost;
  cost.Q = (vector_t(4) << 1, 1, 0.1, 0.1).finished().asDiagonal();
  cost.R = (vector_t(2) << 0.3, 0.3).finished().asDiagonal();
  return cost;
}

inline DynamicsLinearApproximation getFixedDynamics() {
  DynamicsLinearApproximation dynamics;
  // A = [I(2) dt * I(2); zeros(2, 2) I(2)];
  // B = [0.5 * dt * dt * I(2); dt * I(2)];
  scalar_t dt = 0.1;
  dynamics.A = (matrix_t(4, 4) << matrix_t::Identity(2, 2), dt * matrix_t::Identity(2, 2), matrix_t::Zero(2, 2), matrix_t::Identity(2, 2))
                   .finished();
  dynamics.B = (matrix_t(4, 2) << 0.5 * dt * dt * matrix_t::Identity(2, 2), dt * matrix_t::Identity(2, 2)).finished();
  return dynamics;
}

}  // namespace pdal
