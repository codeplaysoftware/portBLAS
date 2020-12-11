
#include "blas_test.hpp"

template<typename scalar_t>
using combination_t = std::tuple<int, int>;

template <typename T>
void transpose(T* A, size_t m, size_t n, size_t lda) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < i; ++j) {
      std::swap(A[j + i * lda], A[i + j * lda]);
    }
  }
}

template <typename T>
void printMatrix(T* A, size_t m, size_t n, size_t lda,
                 bool bIsColMajor = true) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T Element = bIsColMajor ? A[i + j * lda] : A[j + i * lda];
      std::cout << std::setw(4) << Element << " ";
    }
    std::cout << std::endl;
  }
}

template<typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m;
  int n;
  std::tie(m, n) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // clang-format off
  std::vector<data_t> A = {
    1, 0, 0, 0,
    2, 1, 0, 0,
    3, 4, 1, 0,
    5, 6, 7, 1
  };
  std::vector<data_t> B = {
    1, 2, -1, -2,
    1, 2, -1, -2,
    1, 2, -1, -2,
    1, 2, -1, -2
  };
  scalar_t alpha = scalar_t{ 1 };
  int lda = m;
  int ldb = m;
  // clang-format on

  transpose(A.data(), m, n, lda);
  transpose(B.data(), m, n, ldb);

  std::cout << "Matrix A" << std::endl;
  printMatrix(A.data(), m, n, lda);
  std::cout << std::endl;

  std::cout << "Matrix B" << std::endl;
  printMatrix(B.data(), m, n, lda);
  std::cout << std::endl;

  auto q = make_queue();
  test_executor_t ex(q);
  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(A, A.size());
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(B, B.size());

  char side = 'l';
  char transA = 'n';
  char diag = 'u';
  char triangle = 'l';
  _trsm(ex, side, triangle, transA, diag, m, n, alpha, a_gpu, lda, b_gpu, ldb);

  // Temporary printing the solution for inspection
  std::vector<data_t> X(B.size(), data_t{0});
  auto trsmEvent = ex.get_policy_handler().copy_to_host(b_gpu, X.data(), X.size());
  ex.get_policy_handler().wait(trsmEvent);

  std::cout << "Solution" << std::endl;
  printMatrix(X.data(), m, n, ldb);
  std::cout << std::endl;

  // Reset X for proper verification
  X = std::vector<data_t>(B.size(), data_t{0});

  auto x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(X.size());
  if (side == 'l') {
    // Verification of the result. AX = B
    _gemm(ex, transA, transA, m, n, n, data_t{1}, a_gpu, lda, b_gpu, ldb,
          data_t{0}, x_gpu, ldb);
  } else {
    // Verification of the result. XA = B
    _gemm(ex, transA, transA, m, n, n, data_t{1}, b_gpu, ldb, a_gpu, lda,
          data_t{0}, x_gpu, ldb);
  }
  // Copy the verification result to X
  ex.get_policy_handler().wait(
      ex.get_policy_handler().copy_to_host(x_gpu, X.data(), X.size()));

  std::cout << "X must be equal to B" << std::endl;
  printMatrix(X.data(), m, n, ldb);
  std::cout << std::endl;

  ASSERT_TRUE(utils::compare_vectors(X, B));
  ex.get_policy_handler().wait();
}

const auto combi = ::testing::Combine(::testing::Values(4),  // m
                                      ::testing::Values(4)   // n
);

BLAS_REGISTER_TEST(Trsm, combination_t, combi);
