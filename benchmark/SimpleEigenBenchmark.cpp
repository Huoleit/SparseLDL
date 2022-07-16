#include <Eigen/Core>

#include <iostream>

using namespace Eigen;

Matrix<float, 6, 6> __attribute__((noinline)) f() {
  Matrix<float, 6, 6> A, B, C;
  A.setRandom();
  B.setRandom();
  EIGEN_ASM_COMMENT("begin Eigen Test");
  C.noalias() = A * B;
  EIGEN_ASM_COMMENT("begin Eigen Test");
  return C;
}

int main() {
  std::cout << f() << std::endl;
  return 0;
}