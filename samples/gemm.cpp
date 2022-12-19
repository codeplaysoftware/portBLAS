#include "sycl_blas.hpp"
#include <CL/sycl.hpp>

#include "util.hpp"

int main(int argc, char** argv) {
  /* Create a SYCL queue with the default device selector */
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::default_selector());

  /* Create a SYCL-BLAS sb_handle and get the policy handler */
  blas::SB_Handle sb_handle(q);

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
  blas::helper::copy_to_device(sb_handle.get_queue(), A.data(), a_gpu, lda * k);
  blas::helper::copy_to_device(sb_handle.get_queue(), B.data(), b_gpu, ldb * n);
  blas::helper::copy_to_device(sb_handle.get_queue(), C.data(), c_gpu, ldc * n);

  /* Execute the GEMM operation */
  std::cout << "Executing C = " << alpha << "*A*B + " << beta << "*C\n";
  blas::_gemm(sb_handle, 'n', 'n', m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
              c_gpu, ldc);

  /* Copy the result to the host */
  std::cout << "Copying C to host\n";
  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), c_gpu,
                                          C.data(), ldc * n);
  sb_handle.wait(event);

  /* Print the result after the GEMM operation */
  std::cout << "---\nC (after):" << std::endl;
  print_matrix(C, m, n, ldc);

  return 0;
}
