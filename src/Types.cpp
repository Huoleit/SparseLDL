#include "SparseLDL/Types.h"
#include "SparseLDL/implementation/Types.h"

namespace pdal {
// explicit instantiation
template struct tpl::DynamicsLinearApproximation<float>;
template struct tpl::DynamicsLinearApproximation<double>;
template struct tpl::CostApproximation<float>;
template struct tpl::CostApproximation<double>;
}  // namespace pdal
