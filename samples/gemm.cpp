#include "sycl_blas.hpp"
#include <CL/sycl.hpp>

#include <iostream>
#include <random>
#include <string>
#include <vector>

/* Forward-declared utility functions */
template <typename index_t, typename matrix_t>
void print_matrix(const matrix_t& M, index_t rows, index_t cols, index_t ld);
template <typename index_t, typename matrix_t>
void fill_matrix(matrix_t& M, index_t rows, index_t cols, index_t ld);

int main(int argc, char** argv) {
  /* Create a SYCL queue with the default device selector */
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::default_selector());

  /* Create a SYCL-BLAS executor and get the policy handler */
  blas::Executor<blas::PolicyHandler<blas::codeplay_policy>> executor(q);
  auto policy_handler = executor.get_policy_handler();

  /* Arguments of the Gemm operation.
   * Note: these matrix dimensions are too small to get a performance gain by
   * using SYCL-BLAS, but they are convenient for this sample */
  const size_t m = 7;
  const size_t k = 9;
  const size_t n = 5;
  const size_t lda = 12;
  const size_t ldb = 17;
  const size_t ldc = 10;
  const float alpha = 1.5;
  const float beta = 0.5;

  /* Create the matrices */
  std::vector<float> A = std::vector<float>(lda * k);
  std::vector<float> B = std::vector<float>(ldb * n);
  std::vector<float> C = std::vector<float>(ldc * n);

  /* Fill the matrices with random values */
  fill_matrix(A, m, k, lda);
  fill_matrix(B, k, n, ldb);
  fill_matrix(C, m, n, ldc);

  /* Print the matrices before the GEMM operation */
  std::cout << "A:\n";
  print_matrix(A, m, k, lda);
  std::cout << "---\nB:\n";
  print_matrix(B, k, n, ldb);
  std::cout << "---\nC (before):\n";
  print_matrix(C, m, n, ldc);

  /* Create the buffers */
  auto a_gpu = blas::make_sycl_iterator_buffer<float>(lda * k);
  auto b_gpu = blas::make_sycl_iterator_buffer<float>(ldb * n);
  auto c_gpu = blas::make_sycl_iterator_buffer<float>(ldc * n);

  /* Copy the matrices to the device
   * Note: this sample uses explicit copy operations, see the GEMV sample for
   * an alternative way
   */
  std::cout << "---\nCopying A, B and C to device\n";
  policy_handler.copy_to_device(A.data(), a_gpu, lda * k);
  policy_handler.copy_to_device(B.data(), b_gpu, ldb * n);
  policy_handler.copy_to_device(C.data(), c_gpu, ldc * n);

  /* Execute the GEMM operation */
  std::cout << "Executing C = " << alpha << "*A*B + " << beta << "*C\n";
  blas::_gemm(executor, 'n', 'n', m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
              c_gpu, ldc);

  /* Copy the result to the host */
  std::cout << "Copying C to host\n";
  auto event = policy_handler.copy_to_host(c_gpu, C.data(), ldc * n);
  policy_handler.wait(event);

  /* Print the result after the GEMM operation */
  std::cout << "---\nC (after):" << std::endl;
  print_matrix(C, m, n, ldc);

  return 0;
}

template <typename index_t, typename matrix_t>
void print_matrix(const matrix_t& M, index_t rows, index_t cols, index_t ld) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::string print_friendly = std::to_string(M[j * ld + i]).substr(0, 6);
      std::cout << print_friendly << ((j < cols - 1) ? ' ' : '\n');
    }
  }
}

template <typename index_t, typename matrix_t>
void fill_matrix(matrix_t& M, index_t rows, index_t cols, index_t ld) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      M[j * ld + i] = dis(gen);
    }
  }
}
