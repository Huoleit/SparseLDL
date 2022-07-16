#pragma once

#include "SparseLDL/Types.h"

#include <Eigen/LU>

namespace pdal {
template <typename Scalar, int Nx, int Nu>
struct DxCollection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<Scalar, Nu, Nu> D0;
  Eigen::Matrix<Scalar, Nx, Nx> D1;
  Eigen::Matrix<Scalar, Nx, Nx> D2;
  Eigen::Matrix<Scalar, Nu, Nu> D3;
  Eigen::Matrix<Scalar, Nx, Nx> D4;
  Eigen::Matrix<Scalar, Nx, Nx> D5;
};

template <typename Scalar, int Nx, int Nu>
struct LxCollection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<Scalar, Nx, Nu> L10;
  Eigen::Matrix<Scalar, Nx, Nx> L21;
  Eigen::Matrix<Scalar, Nx, Nx> L42;
  Eigen::Matrix<Scalar, Nx, Nu> L43;
  Eigen::Matrix<Scalar, Nx, Nx> L54;
};

template <typename Scalar, int Nx, int Nu>
void solveWithSparseLDL(const DynamicsAlignedStdVector<Scalar, Nx, Nu>& dynamics, const CostAlignedStdVector<Scalar, Nx, Nu>& cost,
                        LxCollection<Scalar, Nx, Nu>& Lx, DxCollection<Scalar, Nx, Nu>& D, DxCollection<Scalar, Nx, Nu>& DInv,
                        vector_s_t<Scalar>& b, Scalar eps = 1e-5) {
  D.D0 = cost[0].R;
  DInv.D0 = D.D0.inverse();
  Lx.L10.noalias() = dynamics[0].B * DInv.D0;

  D.D1.noalias() = -dynamics[0].B * Lx.L10.transpose();
  D.D1 -= eps * Eigen::Matrix<Scalar, Nx, Nx>::Identity();
  DInv.D1 = D.D1.inverse();
  Lx.L21 = -DInv.D1;

  D.D2 = cost[1].Q;
  D.D2 -= DInv.D1;
  DInv.D2 = D.D2.inverse();

  D.D3 = cost[1].R;
  DInv.D3 = D.D3.inverse();
  Lx.L42.noalias() = dynamics[1].A * DInv.D2;
  Lx.L43.noalias() = dynamics[1].B * DInv.D3;

  D.D4.noalias() = -dynamics[1].A * Lx.L42.transpose();
  D.D4.noalias() -= dynamics[1].B * Lx.L43.transpose();
  D.D4 -= eps * Eigen::Matrix<Scalar, Nx, Nx>::Identity();
  DInv.D4 = D.D4.inverse();

  Lx.L54 = -DInv.D4;
  D.D5 = cost[2].Q;
  D.D5 -= DInv.D4;
  DInv.D5 = D.D5.inverse();

  // Solves (L+I)x = b
  b.template segment<Nx>(Nu).noalias() -= Lx.L10 * b.template segment<Nu>(0);
  b.template segment<Nx>(Nu + Nx).noalias() -= Lx.L21 * b.template segment<Nx>(Nu);
  b.template segment<Nx>(Nu + Nx + Nx + Nu).noalias() -= Lx.L42 * b.template segment<Nx>(Nu + Nx);
  b.template segment<Nx>(Nu + Nx + Nx + Nu).noalias() -= Lx.L43 * b.template segment<Nu>(Nu + Nx + Nx);
  b.template segment<Nx>(Nu + Nx + Nx + Nu + Nx).noalias() -= Lx.L54 * b.template segment<Nx>(Nu + Nx + Nx + Nu);

  b.template segment<Nu>(0).transpose() *= DInv.D0;
  b.template segment<Nx>(Nu).transpose() *= DInv.D1;
  b.template segment<Nx>(Nu + Nx).transpose() *= DInv.D2;
  b.template segment<Nu>(Nu + Nx + Nx).transpose() *= DInv.D3;
  b.template segment<Nx>(Nu + Nx + Nx + Nu).transpose() *= DInv.D4;
  b.template segment<Nx>(Nu + Nx + Nx + Nu + Nx).transpose() *= DInv.D5;

  b.template segment<Nx>(Nu + Nx + Nx + Nu).noalias() -= Lx.L54.transpose() * b.template segment<Nx>(Nu + Nx + Nx + Nu + Nx);
  b.template segment<Nu>(Nu + Nx + Nx).noalias() -= Lx.L43.transpose() * b.template segment<Nx>(Nu + Nx + Nx + Nu);
  b.template segment<Nx>(Nu + Nx).noalias() -= Lx.L42.transpose() * b.template segment<Nx>(Nu + Nx + Nx + Nu);
  b.template segment<Nx>(Nu).noalias() -= Lx.L21.transpose() * b.template segment<Nx>(Nu + Nx);
  b.template segment<Nu>(0).noalias() -= Lx.L10.transpose() * b.template segment<Nx>(Nu);
};
}  // namespace pdal
