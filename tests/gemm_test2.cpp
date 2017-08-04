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
#include <utils/vec.hpp>


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
  std::cout << "err = " << relative_diff(refC, dataC) << std::endl; \
}

          // auto A = make_matrix<storage_type::cms>(m, k, accA.get_pointer());
          // auto B = make_matrix<storage_type::cms>(k, n, accB.get_pointer());
          // auto C = make_matrix<storage_type::cms>(m, n, accC.get_pointer());
          // auto S = make_matrix<storage_type::cms>(
          //    _local_rows, _local_cols, scratch.get_pointer()); \

/*
template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v1, GemmV1, "",
    (m*n + lr - 1) / lr,
    _gemm_v1(id.get_global(0), 1.0f, A, B, 1.0f, C))

template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v2, GemmV2, "",
    (m*n + lr - 1) / lr,
    _gemm_v2(id.get_global(0), 1.0f, A, B, 1.0f, C))

template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v3, GemmV3, "",
    ((m + lr - 1) / lr) * n,
    _gemm_v3(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <int> class GemmV4;

template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v4, GemmV4<work>, "work = " << work,
    ((m + lr - 1) / lr) * ((n + work - 1) / work),
    _gemm_v4<work>(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v5, GemmV5, "",
    ((m + lr - 1) / lr) * ((n + 4 - 1) / 4),
    _gemm_v5(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v6, GemmV6, "",
    ((m + lr - 1) / lr) * ((n + 4 - 1) / 4),
    _gemm_v6(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <int> class GemmV7;

template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v7, GemmV7<work>, "work = " << work,
    ((m + lr - 1) / lr) * ((n + work - 1) / work),
    _gemm_v7<work>(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v8, GemmV8, "",
    ((m + lr - 1) / lr) * ((n + 8 - 1) / 8),
    _gemm_v8(id.get_group(0), lr, id.get_local(0), 1.0f, A, B, 1.0f, C))

template <int> class GemmV9;

#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA ,int tpr
template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v9, GemmV9<work>, "work = " << work << ", tpr = " << tpr,
    ((m + (lr/tpr) - 1) / (lr/tpr)) * ((n + (work*tpr) - 1) / (work*tpr)),
    _gemm_v9<work>(id.get_group(0), lr, id.get_local(0), tpr,
                   1.0f, A, B, 1.0f, C))
#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA


template <int> class GemmV10;

#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA ,int tpr
template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST(gemm_v10, GemmV10<work>, "work = " << work << ", tpr = " << tpr,
    ((m + (lr/tpr) - 1) / (lr/tpr)) * ((n + (work*tpr) - 1) / (work*tpr)),
    _gemm_v10<work>(id.get_group(0), lr, id.get_local(0), tpr,
                   1.0f, A, B, 1.0f, C))
#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA


template <int> class GemmV11;

#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA ,int tpr, int ssize
template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v11, GemmV11<work>,
    "work = " << work << ", tpr = " << tpr << ", ssize = " << ssize,
    ((m + (lr/tpr) - 1) / (lr/tpr)) * ((n + (work*tpr) - 1) / (work*tpr)),
    ssize, work*tpr,
    _gemm_v11<work>(id, id.get_group(0), lr, id.get_local(0), tpr,
                    1.0f, A, B, 1.0f, C, S))
#undef ENABLE_TEST_USER_DATA
#define ENABLE_TEST_USER_DATA


template <int> class GemmV12;

template <int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v12, GemmV12<work>,
    "work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    cl_size / sizeof(element_type), work * cl_size / sizeof(element_type),
    _gemm_v12<work>(id, id.get_group(0), lr, id.get_local(0),
                    1.0f, A, B, 1.0f, C, S))


template <int, int> class GemmV13;
#define _tparams rsize, work
template <int rsize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v13, GemmV13<_tparams>,
    "rsize= " << rsize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    cl_size/sizeof(element_type) + 1, work * cl_size/sizeof(element_type),
    _gemm_v13<_tparams>(id, id.get_group(0), lr, id.get_local(0),
                       1.0f, A, B, 1.0f, C, S))
#undef _tparams
*/


/*
template <int, int, int> class GemmV14;
#define _tparams rsize, csize, work
template <int rsize, int csize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v14, GemmV14<_tparams>,
    "rsize = " << rsize << " csize = " << csize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    2*cl_size/sizeof(element_type) + 1, work * cl_size/sizeof(element_type),
    _gemm_v14<_tparams>(id, id.get_group(0), id.get_local(0),
                       1.0f, A, B, 1.0f, C, S))
#undef _tparams
*/

template <int, int, int> class GemmV14;
#define _tparams rsize, csize, work
template <int rsize, int csize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v14, GemmV14<_tparams>,
    "rsize = " << rsize << " csize = " << csize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    2*cl_size/sizeof(element_type) + 1, work * cl_size/sizeof(element_type),
    _gemm_v14<_tparams>(
      id, id.get_group(0), id.get_local(0), m, n, k, element_type(1),
      accA.get_pointer(), m, accB.get_pointer(), k, element_type(1),
      accC.get_pointer(), m, scratch.get_pointer(),
      2*cl_size/sizeof(element_type) + 1))
#undef _tparams


template <int, int, int> class GemmV15;
#define _tparams rsize, csize, work
template <int rsize, int csize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v15, GemmV15<_tparams>,
    "rsize = " << rsize << " csize = " << csize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    4*cl_size/sizeof(element_type), work * cl_size/sizeof(element_type),
    _gemm_v15<_tparams>(
      id, id.get_group(0), id.get_local(0), m, n, k, element_type(1),
      accA.get_pointer(), m, accB.get_pointer(), k, element_type(1),
      accC.get_pointer(), m, scratch.get_pointer()))
#undef _tparams


template <int, int, int> class GemmV16;
#define _tparams rsize, csize, work
template <int rsize, int csize, int work, ENABLE_TEST_PARAMS>
ENABLE_TEST_LOCAL(gemm_v16, GemmV16<_tparams>,
    "rsize = " << rsize << " csize = " << csize << " work = " << work,
    ((m + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))) *
    ((n + work * cl_size / sizeof(element_type) - 1) /
     (work * cl_size / sizeof(element_type))),
    4*cl_size/sizeof(element_type), work * cl_size/sizeof(element_type),
    _gemm_v16<_tparams>(
      id, id.get_group(0), id.get_local(0), m, n, k, element_type(1),
      accA.get_pointer(), m, accB.get_pointer(), k, element_type(1),
      accC.get_pointer(), m, scratch.get_pointer()))
#undef _tparams


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

  cl::sycl::queue q;
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  run_test(5, 2.0*m*n*k, [&] {
    gemm("N", "N", m, n, k, element_type(1), dataA.data(), m, dataB.data(), k,
         element_type(1), refC.data(), m);
  });

  /*
  for (int lr = 64; lr <=256; lr += 64) {
    test_gemm_v1(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v2(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v3(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v4<1>(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v7<1>(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v4<2>(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v7<2>(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v4<4>(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v5(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v6(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v7<4>(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v4<8>(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v7<8>(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v8(lr, m, n, k, dataA, dataB, origC, refC, q);
    // test_gemm_v4<16>(lr, m, n, k, dataA, dataB, origC, refC, q);
    test_gemm_v7<16>(lr, m, n, k, dataA, dataB, origC, refC, q);
    for (int tpr = 1; lr / tpr >= 16; tpr *= 2) {
      test_gemm_v9<1>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      test_gemm_v10<1>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      for (int s = 16; s <= 128; s *= 2) {
        test_gemm_v11<1>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr, s);
      }
      test_gemm_v9<2>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      test_gemm_v10<2>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      for (int s = 16; s <= 128; s *= 2) {
        test_gemm_v11<2>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr, s);
      }
      test_gemm_v9<4>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      test_gemm_v10<4>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      for (int s = 16; s <= 128; s *= 2) {
        test_gemm_v11<4>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr, s);
      }
      test_gemm_v9<8>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      test_gemm_v10<8>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      for (int s = 16; s <= 128; s *= 2) {
        test_gemm_v11<8>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr, s);
      }
      test_gemm_v9<16>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      test_gemm_v10<16>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr);
      for (int s = 16; s <= 128; s *= 2) {
        test_gemm_v11<16>(lr, m, n, k, dataA, dataB, origC, refC, q, tpr, s);
      }
    }
  }
  */

  const int lrm = cl_size / sizeof(element_type);
  /*
  test_gemm_v12<1>(1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v12<2>(2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v12<4>(4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v12<8>(8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v12<16>(16 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  */

  /*
  test_gemm_v13<1, 1>(1 * 1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<1, 2>(1 * 2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<1, 4>(1 * 4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<1, 8>(1 * 8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<1, 16>(1 * 16 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  test_gemm_v13<2, 2>(2 * 2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<2, 4>(2 * 4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v13<2, 8>(2 * 8 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  test_gemm_v13<4, 4>(4 * 4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  */

  //test_gemm_v14<1, 1, 1>(1*1*1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<1, 1, 2>(1*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<1, 1, 4>(1*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<1, 1, 4>(1*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<1, 1, 4>(1*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<1, 1, 4>(1*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<1, 1, 8>(1*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v15<1, 1, 8>(1*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v16<1, 1, 8>(1*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v14<1, 1, 16>(1*1*16 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  //test_gemm_v14<2, 1, 2>(2*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<2, 1, 2>(2*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<2, 1, 2>(2*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 1, 2>(2*1*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<2, 1, 4>(2*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<2, 1, 4>(2*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<2, 1, 4>(2*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 1, 4>(2*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<2, 1, 8>(2*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<2, 1, 8>(2*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<2, 1, 8>(2*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 1, 8>(2*1*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  //test_gemm_v14<4, 1, 4>(4*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<4, 1, 4>(4*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<4, 1, 4>(4*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<4, 1, 4>(4*1*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);


  // test_gemm_v14<1, 2, 1>(1*2*1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v14<1, 2, 2>(1*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<1, 2, 4>(1*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<1, 2, 4>(1*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<1, 2, 4>(1*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<1, 2, 4>(1*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<1, 2, 8>(1*2*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<1, 2, 8>(1*2*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<1, 2, 8>(1*2*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<1, 2, 8>(1*2*8 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  //test_gemm_v14<2, 2, 2>(2*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<2, 2, 2>(2*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<2, 2, 2>(2*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 2, 2>(2*2*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v14<2, 2, 4>(2*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  //test_gemm_v15<2, 2, 4>(2*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v16<2, 2, 4>(2*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  test_gemm_v17<2, 2, 4>(2*2*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);


  // test_gemm_v14<1, 4, 1>(1*4*1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v14<1, 4, 2>(1*4*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v14<1, 4, 4>(1*4*4 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  // test_gemm_v14<2, 4, 2>(2*4*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);


  // test_gemm_v14<1, 8, 1>(1*8*1 * lrm, m, n, k, dataA, dataB, origC, refC, q);
  // test_gemm_v14<1, 8, 2>(1*8*2 * lrm, m, n, k, dataA, dataB, origC, refC, q);


  // test_gemm_v14<1, 16, 1>(1*16*1 * lrm, m, n, k, dataA, dataB, origC, refC, q);

  return 0;
}

