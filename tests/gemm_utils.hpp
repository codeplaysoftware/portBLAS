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
 *  @filename gemm_utils.hpp
 *
 **************************************************************************/


#include <iostream>
#include <random>
#include <vector>


#include <CL/sycl.hpp>


#include <interface/blas3_interface_sycl_gemm.hpp>
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


template <typename T> class Test {};


template <typename Gemm, typename T, typename Container>
typename std::enable_if<Gemm::version == 2>::type
test(int r, int m, int n, int k, T alpha, const Container &dataA, int lda,
     const Container &dataB, int ldb, T beta, Container dataC, int ldc,
     const Container &refC, cl::sycl::queue q)
{
  using etype = typename Gemm::value_type;
  std::cout << "\n=== Testing " << Gemm::get_type_string() << " ==="
            << std::endl;
  {
    buffer<etype, 1> buffA(dataA.data(), range<1>(dataA.size()));
    buffer<etype, 1> buffB(dataB.data(), range<1>(dataB.size()));
    buffer<etype, 1> buffC(dataC.data(), range<1>(dataC.size()));
    run_test(r, 2.0*m*n*k, [&] {
      q.submit([&] (handler &cgh) {
        auto accA = buffA.template get_access<access::mode::read>(cgh);
        auto accB = buffB.template get_access<access::mode::read>(cgh);
        auto accC = buffC.template get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<Test<Gemm>>(Gemm::get_nd_range(m, n),
            [=](nd_item<1> id) {
          Gemm::run(id.get_global(0), m, n, k, etype(alpha),
                    accA.get_pointer(), lda, accB.get_pointer(), ldb,
                    etype(beta), accC.get_pointer(), ldc);
        });
      });
      q.wait();
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}


template <typename Gemm, typename T, typename Container>
typename std::enable_if<Gemm::version == 19>::type
test(int r, int m, int n, int k, T alpha, const Container &dataA, int lda,
     const Container &dataB, int ldb, T beta, Container dataC, int ldc, 
     const Container &refC, cl::sycl::queue q)
{
  using etype = typename Gemm::value_type;
  std::cout << "\n=== Testing " << Gemm::get_type_string() << " ==="
            << std::endl;
  {
    buffer<etype, 1> buffA(dataA.data(), range<1>(dataA.size()));
    buffer<etype, 1> buffB(dataB.data(), range<1>(dataB.size()));
    buffer<etype, 1> buffC(dataC.data(), range<1>(dataC.size()));
    run_test(r, 2.0*m*n*k, [&] {
      q.submit([&] (handler &cgh) {
        auto accA = buffA.template get_access<access::mode::read>(cgh);
        auto accB = buffB.template get_access<access::mode::read>(cgh);
        auto accC = buffC.template get_access<access::mode::read_write>(cgh);
        accessor<etype, 1, access::mode::read_write, access::target::local>
          scratch(range<1>(Gemm::scratch_size), cgh);
        cgh.parallel_for<Test<Gemm>>(Gemm::get_nd_range(m, n),
            [=](nd_item<1> id) {
          Gemm::run(
              id, id.get_group(0), id.get_local(0), m, n, k, etype(alpha),
              accA.get_pointer(), lda, accB.get_pointer(), ldb, etype(beta),
              accC.get_pointer(), ldc, scratch.get_pointer());
        });
      });
      q.wait();
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}


template <typename T, typename Container>
void test_syclblas(int r, char transA, char transB, int m, int n, int k,
                   T alpha, const Container &dataA, int lda,
                   const Container &dataB, int ldb, T beta,
                   Container dataC, int ldc, const Container &refC,
                   cl::sycl::queue q) {
  using etype = typename Container::value_type;
  std::cout << "\n=== Testing SYCL-BLAS gemm ==="
            << std::endl;
  Executor<SYCL> ex(q);
  {
    buffer<etype, 1> buffA(dataA.data(), range<1>(dataA.size()));
    buffer<etype, 1> buffB(dataB.data(), range<1>(dataB.size()));
    buffer<etype, 1> buffC(dataC.data(), range<1>(dataC.size()));
    run_test(r, 2.0*m*n*k, [&] {
      _gemm(ex, transA, transB, m, n, k, alpha,
            buffA, lda, buffB, ldb, beta, buffC, ldc);
      q.wait();
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}


template <bool TransA, bool TransB, typename E>
void run_gemm_tests(int seed, int m, int k, int n, int rep)
{
  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<E>(TransA ? k : m, TransA ? m : k, -1, 1, rnd);
  auto dataB = gen_matrix<E>(TransB ? n : k, TransB ? k : n, -1, 1, rnd);
  auto origC = gen_matrix<E>(m, n, -1, 1, rnd);
  auto refC = origC;

  const char *ta_str = TransA ? "T" : "N";
  const char *tb_str = TransB ? "T" : "N";

  const int lda = TransA ? k : m;
  const int ldb = TransB ? n : k;
  const int ldc = m;

  std::cout << "\n=== Testing system CPU implementation ===" << std::endl;
  run_test(rep, 2.0*m*n*k, [&] {
    gemm(ta_str, tb_str, m, n, k, E(1), dataA.data(), lda, dataB.data(),
         ldb, E(1), refC.data(), m);
  });

  cl::sycl::queue q([=](cl::sycl::exception_list eL) {
    try {
      for (auto &e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception &e) {
      std::cout << " E " << e.what() << std::endl;
    } catch (...) {
      std::cout << " An exception " << std::endl;
    }
  });

  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

#define ARG m, n, k, E(1), dataA, lda, dataB, ldb, E(1), origC, \
            ldc, refC, q

  const int cls = 64; // size of cache line in bytes
  const bool db = false; // use double buffer
  const bool ba = false; // avoid bank conflicts for A
  const bool bb = false; // avoid bank conflicts for B
  const bool ta = TransA;
  const bool tb = TransB;

  test<ReferenceGemmFactory<128, ta, tb, E>>(rep, ARG);

#define TARG(_tir, _tic, _twr, _twc, _ttr, _ttc) \
    GemmFactory<db, ba, bb, cls, Tile<_tir, _tic, _twr, _twc, _ttr, _ttc>, \
                ta, tb, E>
  test<TARG(8, 8, 8, 8, 1, 1)>(rep, ARG);
  test<TARG(8, 8, 8, 8, 2, 2)>(rep, ARG);
  test<TARG(8, 8, 8, 8, 4, 4)>(rep, ARG);
  test<TARG(8, 8, 8, 8, 8, 8)>(rep, ARG);
  test<TARG(8, 8, 8, 8, 16, 16)>(rep, ARG);

  test<TARG(8, 8, 16, 16, 1, 1)>(rep, ARG);
  test<TARG(8, 8, 16, 16, 2, 2)>(rep, ARG);
  test<TARG(8, 8, 16, 16, 4, 4)>(rep, ARG);
  test<TARG(8, 8, 16, 16, 8, 8)>(rep, ARG);
  test<TARG(8, 8, 16, 16, 16, 16)>(rep, ARG);

#undef TARG

  test_syclblas(rep, *ta_str, *tb_str, ARG);

#undef ARG

}

