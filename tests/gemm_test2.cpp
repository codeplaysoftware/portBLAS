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
                    _command) \
void test_##_name(int lr, int m, int n, int k, const Container &dataA, \
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
    run_test(5, 2.0*m*n*k, [&] { \
      q.submit([&] (cl::sycl::handler &cgh) { \
        auto accA = \
            buffA.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accB = \
            buffB.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accC = buffC.template get_access< \
              cl::sycl::access::mode::read_write>(cgh); \
        const cl::sycl::range<1> local_range = lr; \
        const cl::sycl::range<1> global_range = (_global_range_expr); \
        cgh.parallel_for<class _kernel_name>( \
            cl::sycl::nd_range<1>(global_range*local_range, local_range), \
            [=](cl::sycl::nd_item<1> id) { \
          auto A = make_matrix<storage_type::cms>(m, k, accA); \
          auto B = make_matrix<storage_type::cms>(k, n, accB); \
          auto C = make_matrix<storage_type::cms>(m, n, accC); \
          _command; \
        }); \
      }); \
      q.wait(); \
    }); \
  } \
  std::cout << "err = " << relative_diff(refC, dataC) << std::endl; \
}

#define ENABLE_TEST_LOCAL(_name, _kernel_name, _desc_data, \
                          _global_range_expr, _local_rows, _local_cols, \
                          _command) \
void test_##_name(int lr, int m, int n, int k, const Container &dataA, \
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
    run_test(5, 2.0*m*n*k, [&] { \
      q.submit([&] (cl::sycl::handler &cgh) { \
        auto accA = \
            buffA.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accB = \
            buffB.template get_access<cl::sycl::access::mode::read>(cgh); \
        auto accC = buffC.template get_access< \
              cl::sycl::access::mode::read_write>(cgh); \
        cl::sycl::accessor<element_type, 1, \
            cl::sycl::access::mode::read_write, access::target::local> \
          scratch(cl::sycl::range<1>((_local_rows)*(_local_cols)), cgh); \
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


template <int, int, int> class GemmV17;
template <ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v2, GemmV2, "",
    (m*n - 1) / lr + 1, 4, 1,
    _gemm_v2(
      id.get_global(0), m, n, k, element_type(1), accA.get_pointer(), m,
      accB.get_pointer(), k, element_type(1), accC.get_pointer(), m))


template <int, int, int> class GemmV17;
#define _tparams rsize, csize, work
template <int rsize, int csize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v17, GemmV17<_tparams>,
    "rsize = " << rsize << " csize = " << csize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    4*cl_size/sizeof(element_type), work * cl_size/sizeof(element_type),
    _gemm_v17<_tparams>(
      id, id.get_group(0), id.get_local(0), m, n, k, element_type(1),
      accA.get_pointer(), m, accB.get_pointer(), k,  element_type(1),
      accC.get_pointer(), m, scratch.get_pointer()))
#undef _tparams


#undef ENABLE_TEST_USER_DATA
#undef ENABLE_TEST_PARAMS
#undef ENABLE_TEST


int main(int argc, char *argv[]) {
  using element_type = float;
  const int seed = 42;

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
    return -1;
  }

  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);

  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<element_type>(m, k, -1, 1, rnd);
  auto dataB = gen_matrix<element_type>(k, n, -1, 1, rnd);
  auto origC = gen_matrix<element_type>(m, n, -1, 1, rnd);
  auto refC = origC;


  std::cout << "\n=== Testing system CPU implementation ===" << std::endl;
  run_test(5, 2.0*m*n*k, [&] {
    gemm("N", "N", m, n, k, element_type(1), dataA.data(), m, dataB.data(), k,
         element_type(1), refC.data(), m);
  });

  const int lrm = cl_size / sizeof(element_type);

  cl::sycl::queue q;
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;


  test_gemm_v2(128, m, n, k, dataA, dataB, origC, refC, q);


  test_gemm_v17<1, 1, 4>(1*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  test_gemm_v17<2, 1, 2>(2*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 1, 4>(2*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 1, 8>(2*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  test_gemm_v17<4, 1, 4>(4*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);


  test_gemm_v17<1, 2, 4>(1*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<1, 2, 8>(1*2*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  test_gemm_v17<2, 2, 2>(2*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 2, 4>(2*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  return 0;
}

