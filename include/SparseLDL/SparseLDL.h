#pragma once

#include <vector>

#include "SparseLDL/Types.h"

namespace pdal {
void sparseLDL(const std::vector<DynamicsLinearApproximation>& dynamics, const std::vector<CostApproximation>& cost,
               std::vector<matrix_t>& Lx, std::vector<matrix_t>& DInv, std::vector<matrix_t>& D);

void Lsolve(const std::vector<matrix_t>& Lx, vector_t& b);
void Ltsolve(const std::vector<matrix_t>& Lx, vector_t& b);
void solve(const std::vector<matrix_t>& Lx, const std::vector<matrix_t>& DInv, vector_t& b);

}  // namespace pdal
