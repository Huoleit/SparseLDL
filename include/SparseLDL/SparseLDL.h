#pragma once

#include "SparseLDL/Types.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>

namespace pdal {
template <typename Scalar, int Nx, int Nu>
void sparseLDL(const DynamicsAlignedStdVector<Scalar, Nx, Nu>& dynamics, const CostAlignedStdVector<Scalar, Nx, Nu>& cost,
               std::vector<matrix_s_t<Scalar>>& Lx, std::vector<matrix_s_t<Scalar>>& DInv, std::vector<matrix_s_t<Scalar>>& D,
               Scalar eps = 1e-5) {
  const int N = dynamics.size();  // number of stages

  int nnzL = N * 3;  // u, lambda, x

  Lx.clear();
  D.clear();
  DInv.clear();

  Lx.reserve(nnzL);
  D.reserve(nnzL);
  DInv.reserve(nnzL);

  auto checkAndInverse = [](const std::vector<matrix_s_t<Scalar>>& D, std::vector<matrix_s_t<Scalar>>& DInv) {
    DInv.push_back(D.back().ldlt().solve(matrix_s_t<Scalar>::Identity(D.back().rows(), D.back().rows())));
  };

  for (int k = 0; k < N; ++k) {
    const auto& R = cost[k].R;
    const auto& A = dynamics[k].A;
    const auto& B = dynamics[k].B;
    const auto& QNext = cost[k + 1].Q;

    // D = R
    D.push_back(R);
    checkAndInverse(D, DInv);

    if (k != 0) {
      // Lx = A * inv(D)
      Lx.push_back(A * DInv[DInv.size() - 2]);
    }
    // Lx = B * inv(D)
    Lx.push_back(B * DInv.back());

    // D = -B * Lx'
    D.push_back(-B * Lx.back().transpose());

    if (k != 0) {
      D.back().noalias() -= A * Lx[Lx.size() - 2].transpose();
    }
    D.back().noalias() -= eps * matrix_s_t<Scalar>::Identity(D.back().rows(), D.back().rows());  // regularization
    checkAndInverse(D, DInv);
    // Lx = -D
    Lx.push_back(-DInv.back());

    // D = Q - inv(D)
    D.push_back(QNext - DInv.back());
    checkAndInverse(D, DInv);
  }
};

// Solves (L+I)x = b
template <typename Scalar>
void Lsolve(const std::vector<matrix_s_t<Scalar>>& Lx, vector_s_t<Scalar>& b) {
  if ((Lx.size() + 1) % 3 != 0) {
    throw std::runtime_error("[Lsolve] The size of Lx is wrong.");
  }
  const int N = (Lx.size() + 1) / 3;

  int curRow = Lx[0].cols();
  int curSize = Lx[0].rows();
  int subtrahendRow = 0;
  int subtrahendSize = Lx[0].cols();
  b.segment(curRow, curSize).noalias() -= Lx[0] * b.segment(subtrahendRow, subtrahendSize);

  curRow += curSize;
  curSize = Lx[1].rows();
  subtrahendRow += subtrahendSize;
  subtrahendSize = Lx[1].cols();
  b.segment(curRow, curSize).noalias() -= Lx[1] * b.segment(subtrahendRow, subtrahendSize);

  for (int i = 1; i < N; ++i) {
    curRow += curSize + Lx[3 * i].cols();
    curSize = Lx[3 * i - 1].rows();
    subtrahendRow += subtrahendSize;
    subtrahendSize = Lx[3 * i - 1].cols();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i - 1] * b.segment(subtrahendRow, subtrahendSize);

    subtrahendRow += subtrahendSize;
    subtrahendSize = Lx[3 * i].cols();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i] * b.segment(subtrahendRow, subtrahendSize);

    curRow += curSize;
    curSize = Lx[3 * i + 1].rows();
    subtrahendRow += subtrahendSize;
    subtrahendSize = Lx[3 * i + 1].cols();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i + 1] * b.segment(subtrahendRow, subtrahendSize);
  }
}

// Solves (L+I)'x = b
template <typename Scalar>
void Ltsolve(const std::vector<matrix_s_t<Scalar>>& Lx, vector_s_t<Scalar>& b) {
  if ((Lx.size() + 1) % 3 != 0) {
    throw std::runtime_error("[Lsolve] The size of Lx is wrong.");
  }
  const int N = (Lx.size() + 1) / 3;

  int curRow = 0;
  for (int i = 0; i < Lx.size(); ++i) {
    curRow += Lx[i].cols();
  }
  int curSize;
  int subtrahendRow = curRow + Lx.back().rows();
  int subtrahendSize;

  for (int i = N - 1; i > 0; --i) {
    curRow -= Lx[3 * i + 1].cols();
    curSize = Lx[3 * i + 1].cols();
    subtrahendRow -= Lx[3 * i + 1].rows();
    subtrahendSize = Lx[3 * i + 1].rows();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i + 1].transpose() * b.segment(subtrahendRow, subtrahendSize);

    curRow -= Lx[3 * i].cols();
    curSize = Lx[3 * i].cols();
    subtrahendRow -= Lx[3 * i].rows();
    subtrahendSize = Lx[3 * i].rows();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i].transpose() * b.segment(subtrahendRow, subtrahendSize);

    curRow -= Lx[3 * i - 1].cols();
    curSize = Lx[3 * i - 1].cols();
    b.segment(curRow, curSize).noalias() -= Lx[3 * i - 1].transpose() * b.segment(subtrahendRow, subtrahendSize);

    subtrahendRow -= Lx[3 * i].cols();
  }

  curRow -= Lx[1].cols();
  curSize = Lx[1].cols();
  subtrahendRow -= Lx[1].rows();
  subtrahendSize = Lx[1].rows();
  b.segment(curRow, curSize).noalias() -= Lx[1].transpose() * b.segment(subtrahendRow, subtrahendSize);

  curRow -= Lx[0].cols();
  curSize = Lx[0].cols();
  subtrahendRow -= Lx[0].rows();
  subtrahendSize = Lx[0].rows();
  b.segment(curRow, curSize).noalias() -= Lx[0].transpose() * b.segment(subtrahendRow, subtrahendSize);
};

template <typename Scalar>
void solve(const std::vector<matrix_s_t<Scalar>>& Lx, const std::vector<matrix_s_t<Scalar>>& DInv, vector_s_t<Scalar>& b) {
  vector_s_t<Scalar> tmp = b;
  Lsolve(Lx, tmp);
  int curRow = 0;
  for (int i = 0; i < DInv.size(); ++i) {
    b.segment(curRow, DInv[i].rows()).noalias() = DInv[i] * tmp.segment(curRow, DInv[i].rows());
    curRow += DInv[i].rows();
  }
  Ltsolve(Lx, b);
};
}  // namespace pdal
