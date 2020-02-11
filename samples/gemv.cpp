#include "sycl_blas.hpp"
#include <CL/sycl.hpp>

#include "util.hpp"

int main(int argc, char** argv) {
  /* Create a SYCL queue with the default device selector */
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::default_selector());

  /* Create a SYCL-BLAS executor and get the policy handler */
  blas::Executor<blas::PolicyHandler<blas::codeplay_policy>> executor(q);
  auto policy_handler = executor.get_policy_handler();

  /* Arguments of the Gemm operation.
   * Note: these matrix dimensions are too small to get a performance gain by
   * using SYCL-BLAS, but they are convenient for this sample */
  const int m = 7;
  const int n = 7;
  const int lda = 12;
  const int incx = 2;
  const int incy = 3;
  const float alpha = 1.5;
  const float beta = 0.5;

  /* Create the matrix and vectors */
  const int lx = (n - 1) * incx + 1;
  const int ly = (m - 1) * incy + 1;
  std::vector<float> A = std::vector<float>(lda * n);
  std::vector<float> X = std::vector<float>(lx);
  std::vector<float> Y = std::vector<float>(ly);

  /* Fill the matrices with random values */
  fill_matrix(A, m, n, lda);
  fill_vector(X, n, incx);
  fill_vector(Y, m, incy);

  /* Print the matrices before the GEMV operation */
  std::cout << "A:\n";
  print_matrix(A, m, n, lda);
  std::cout << "---\nX:\n";
  print_vector(X, n, incx);
  std::cout << "---\nY (before):\n";
  print_vector(Y, m, incy);

  /* Execute the GEMV operation
   * Note: you can also use explicit copies, see the GEMM sample
   */
  std::cout << "---\nExecuting Y = " << alpha << "*A*X + " << beta << "*Y\n";
  {
    auto a_gpu = blas::make_sycl_iterator_buffer<float>(A, lda * n);
    auto x_gpu = blas::make_sycl_iterator_buffer<float>(X, lx);
    auto y_gpu = blas::make_sycl_iterator_buffer<float>(Y, ly);
    auto event = blas::_gemv(executor, 'n', m, n, alpha, a_gpu, lda, x_gpu,
                             incx, beta, y_gpu, incy);
  }

  /* Print the result after the GEMM operation */
  std::cout << "---\nY (after):" << std::endl;
  print_vector(Y, m, incy);

  return 0;
}
