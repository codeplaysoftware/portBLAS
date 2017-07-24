#include <iostream>

#include <utils/matrix.hpp>

int main() {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.,
                  11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
                  21., 22., 23., 24., 25., 26., 27., 28., 29., 30.};

  std::cout << std::scientific;

  // interpert data as 5-by-6 column major matrix.
  auto M = blas::make_matrix<blas::storage_type::cms>(5, 6, data);

  std::cout << "The whole matrix:\n" << M << std::endl;
  std::cout << "M(2, 3): " << M(2, 3) << std::endl;
  std::cout << "M(1:3, 4:6):\n" << M(blas::range(1, 3), blas::range(4, 6))
            << std::endl;

  auto subM = M(blas::range(1, 3), blas::range(4, 6));

  std::cout << "M(1:3, 4:6)(0:2, 1):\n" << subM(blas::range(0, 2), 1)
            << std::endl;

  std::cout << "List of 2x2 matrix blocks:\n"
            << "==========================\n";
  for (auto i = blas::range(0, 2); 5 > i; ++i) {
    for (auto j = blas::range(0, 2); 6 > j; ++j) {
      std::cout << "M(" << i << ", " << j << ") =\n"
                << M(i, j) << "\n";
    }
  }
  std::cout << "==========================\n" << std::endl;

  std::cout << "List of overlapped 4x4 matrix blocks:\n"
            << "=====================================\n";
  for (auto i = blas::range(0, 2); 5 > i; ++i) {
    for (auto j = blas::range(0, 2); 6 > j; ++j) {
      std::cout << "M(" << i*2 << ", " << j*2 << ") =\n"
                << M(i*2, j*2) << "\n";
    }
  }
  std::cout << "=====================================" << std::endl;

  std::cout << "Copy block (0:2, 0:2) to (2:4, 2:4), "
            << "set block (0:2, 2:4) to 5:\n";

  M(blas::range(2, 4), blas::range(2, 4)) =
      M(blas::range(0, 2), blas::range(0, 2));

  M(blas::range(0, 2), blas::range(2, 4)) = 5;

  std::cout << M << std::endl;

  return 0;
}

