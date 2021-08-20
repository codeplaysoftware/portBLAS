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
#ifdef SYCL_BLAS_USE_USM
  float* a_gpu = cl::sycl::malloc_device<float>(lda * k, q);
  float* b_gpu = cl::sycl::malloc_device<float>(ldb * n, q);
  float* c_gpu = cl::sycl::malloc_device<float>(ldc * n, q);

  q.memcpy(a_gpu, A.data(), sizeof(float) * lda * k).wait();
  q.memcpy(b_gpu, B.data(), sizeof(float) * ldb * n).wait();
  q.memcpy(c_gpu, C.data(), sizeof(float) * ldc * n).wait();
#else
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
#endif

  /* Execute the GEMM operation */
  std::cout << "Executing C = " << alpha << "*A*B + " << beta << "*C\n";
  auto ev = blas::_gemm(executor, 'n', 'n', m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
              c_gpu, ldc);
#ifdef SYCL_BLAS_USE_USM
  policy_handler.wait(ev);
#endif

  /* Copy the result to the host */
  std::cout << "Copying C to host\n";
  auto event = 
#ifdef SYCL_BLAS_USE_USM
      q.memcpy(C.data(), c_gpu, sizeof(float) * ldc * n);
#else
      policy_handler.copy_to_host(c_gpu, C.data(), ldc * n);
#endif
  policy_handler.wait({event});

  /* Print the result after the GEMM operation */
  std::cout << "---\nC (after):" << std::endl;
  print_matrix(C, m, n, ldc);

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(a_gpu, q);
  cl::sycl::free(b_gpu, q);
  cl::sycl::free(c_gpu, q);
#endif

  return 0;
}
