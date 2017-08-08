/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename gemm_test2.hpp
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <vector>

#include <CL/sycl.hpp>


#include <operations/blas3_trees_gemm.hpp>


#include "test_utils.hpp"


using namespace cl::sycl;
using namespace blas;


#define ENABLE_SYSTEM_GEMM(_type, _system_name) \
extern "C" void _system_name( \
    const char *, const char *, const int *, const int *, const int *, \
    const _type *, const _type *, const int *, const _type *, const int *, \
    const _type *, _type *, const int *); \
void gemm( \
    const char *transA, const char *transB, int m, int n, int k, _type alpha, \
    const _type a[], int lda, const _type b[], int ldb, _type beta, \
    _type c[], int ldc) { \
  _system_name(transA, transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, \
               c, &ldc); \
}

ENABLE_SYSTEM_GEMM(float, sgemm_)
ENABLE_SYSTEM_GEMM(double, dgemm_)

#undef ENABLE_SYSTEM_GEMM


#define ENABLE_TEST_USER_DATA
#define ENABLE_TEST_PARAMS typename Container


#define ENABLE_TEST(_name, _kernel_name, _desc_data, _global_range_expr, \
                    _local_mem_size, _command) \
void test_##_name(int lr, int r, int m, int n, int k, const Container &dataA, \
                  const Container &dataB, Container dataC, \
                  const Container &refC, cl::sycl::queue q \
                  ENABLE_TEST_USER_DATA) { \
  using element_type = typename Container::value_type; \
  std::cout << "\n=== Testing " #_kernel_name "(lr = " << lr << ") "\
            << _desc_data << " ===" << std::endl; \
  { \
    cl::sycl::buffer<element_type, 1> buffA( \
        dataA.data(), cl::sycl::range<1>(dataA.size())); \
    cl::sycl::buffer<element_type, 1> buffB( \
        dataB.data(), cl::sycl::range<1>(dataB.size())); \
    cl::sycl::buffer<element_type, 1> buffC( \
        dataC.data(), cl::sycl::range<1>(dataC.size())); \
    run_test(r, 2.0*m*n*k, [&] { \
      q.submit([&] (cl::sycl::handler &cgh) { \
        auto accA = \
            buffA.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accB = \
            buffB.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accC = buffC.template get_access< \
              cl::sycl::access::mode::read_write>(cgh); \
        cl::sycl::accessor<element_type, 1, \
            cl::sycl::access::mode::read_write, access::target::local> \
          scratch(cl::sycl::range<1>(_local_mem_size), cgh); \
        const cl::sycl::range<1> local_range = lr; \
        const cl::sycl::range<1> global_range = (_global_range_expr); \
        cgh.parallel_for<class _kernel_name>( \
            cl::sycl::nd_range<1>(global_range*local_range, local_range), \
            [=](cl::sycl::nd_item<1> id) { \
          _command; \
        }); \
      }); \
      q.wait(); \
    }); \
  } \
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl; \
}


template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v2, GemmV2, "",
    (m*n - 1) / lr + 1, 1,
    _gemm_v2(
      id.get_global(0), m, n, k, element_type(1), accA.get_pointer(), m,
      accB.get_pointer(), k, element_type(1), accC.get_pointer(), m))


template <bool, int, int, int, int, int> class GemmV19;
#define _tparams double_buffer, cl, item_rows, item_cols, wg_rows, wg_cols
template <bool double_buffer, int cl, int item_rows, int item_cols,
          int wg_rows, int wg_cols, ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v19, GemmV19<_tparams>,
    "item_dim = (" << item_rows << ", " << item_cols << "); " <<
    "wg_dim = (" << wg_rows << ", " << wg_cols << "); " <<
    "db = " << double_buffer,
    ((m - 1)/(wg_rows * item_rows) + 1) * ((n - 1)/(wg_cols * item_cols) + 1),
    (double_buffer+1) * cl/sizeof(element_type) *
    (wg_rows*item_rows + wg_cols*item_cols),
    _gemm_v19<_tparams>(
      id, id.get_group(0), id.get_local(0), m, n, k, element_type(1),
      accA.get_pointer(), m, accB.get_pointer(), k, element_type(1),
      accC.get_pointer(), m, scratch.get_pointer()))


#undef _tparams


#undef ENABLE_TEST_USER_DATA
#undef ENABLE_TEST_PARAMS
#undef ENABLE_TEST


int main(int argc, char *argv[]) {
  using element_type = float;
  const int seed = 42;

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " M N K rep" << std::endl;
    return -1;
  }

  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);
  const int rep = std::atoi(argv[4]);

  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<element_type>(m, k, -1, 1, rnd);
  auto dataB = gen_matrix<element_type>(k, n, -1, 1, rnd);
  auto origC = gen_matrix<element_type>(m, n, -1, 1, rnd);
  auto refC = origC;


  std::cout << "\n=== Testing system CPU implementation ===" << std::endl;
  run_test(rep, 2.0*m*n*k, [&] {
    gemm("N", "N", m, n, k, element_type(1), dataA.data(), m, dataB.data(), k,
         element_type(1), refC.data(), m);
  });

  const int cl = 64;
  const int lrm = cl / sizeof(element_type);

  cl::sycl::queue q;
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;


#define DATA rep, m, n, k, dataA, dataB, origC, refC, q

  test_gemm_v2(128, DATA);

  test_gemm_v19<true, cl, 4, 16, 16, 4>(16*4, DATA);
  test_gemm_v19<false, cl, 4, 16, 16, 4>(16*4, DATA);

  test_gemm_v19<true, cl, 1, 16, 32, 2>(32*2, DATA);
  test_gemm_v19<false, cl, 1, 16, 32, 2>(32*2, DATA);
  test_gemm_v19<true, cl, 2, 16, 32, 4>(32*4, DATA);
  test_gemm_v19<false, cl, 2, 16, 32, 4>(32*4, DATA);
  test_gemm_v19<true, cl, 4, 16, 32, 8>(32*8, DATA);
  test_gemm_v19<false, cl, 4, 16, 32, 8>(32*8, DATA);

  test_gemm_v19<true, cl, 1, 16, 64, 4>(64*4, DATA);
  test_gemm_v19<false, cl, 1, 16, 64, 4>(64*4, DATA);

  test_gemm_v19<true, cl, 4, 8, 16, 8>(16*8, DATA);
  test_gemm_v19<false, cl, 4, 8, 16, 8>(16*8, DATA);
  test_gemm_v19<true, cl, 8, 8, 16, 16>(16*16, DATA);
  test_gemm_v19<false, cl, 8, 8, 16, 16>(16*16, DATA);

  // test_gemm_v19<true, cl, 8, 9, 16, 16>(16*16, DATA);
  test_gemm_v19<false, cl, 8, 9, 16, 16>(16*16, DATA);

  test_gemm_v19<true, cl, 1, 8, 32, 4>(32*4, DATA);
  test_gemm_v19<false, cl, 1, 8, 32, 4>(32*4, DATA);
  test_gemm_v19<true, cl, 2, 8, 32, 8>(32*8, DATA);
  test_gemm_v19<false, cl, 2, 8, 32, 8>(32*8, DATA);

#undef DATA

  return 0;
}

