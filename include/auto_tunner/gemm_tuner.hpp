/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename gemm_tuner.hpp
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <vector>

#include <cmath>
#include <functional>
#include <numeric>

#include <interface/blas3_interface.hpp>

template <typename T, typename RndEngine>
std::vector<T> gen_matrix(int m, int n, T lo, T hi, RndEngine rnd) {
  std::uniform_real_distribution<T> dst(lo, hi);
  std::vector<T> v(m * n);
  for (auto &e : v) {
    e = dst(rnd);
  }
  return v;
}

template <typename T>
T relative_diff(const std::vector<T> &ref, const std::vector<T> &obt) {
  T mag(0);
  for (auto x : ref) {
    mag += x * x;
  }
  T diff = std::inner_product(std::begin(ref), std::end(ref), std::begin(obt),
                              T(0), std::plus<T>(),
                              [](T x, T y) { return (x - y) * (x - y); });
  return std::sqrt(diff / mag);
}

template <typename T>
struct type_name {
  constexpr static const char *const name = "unknown";
};

#define ENABLE_TYPE_NAME(_type)                       \
  template <>                                         \
  struct type_name<_type> {                           \
    constexpr static const char *const name = #_type; \
  };

ENABLE_TYPE_NAME(int)
ENABLE_TYPE_NAME(float)
ENABLE_TYPE_NAME(double)

template <int...>
struct static_list {};

template <typename TestOperator>
void run_test(int rep, double flop_cnt, TestOperator op = TestOperator()) {
  // warmup
  op();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < rep; ++i) {
    op();
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> sec_d = end - start;
  double sec = sec_d.count() / rep;
  std::cout << "time = " << sec * 1e3 << " ms\n"
            << "perf = " << flop_cnt / sec / 1e9 << " GFLOPS" << std::endl;
}

using namespace cl::sycl;
using namespace blas;

#define ENABLE_SYSTEM_GEMM(_type, _system_name)                               \
  extern "C" void _system_name(                                               \
      const char *, const char *, const int *, const int *, const int *,      \
      const _type *, const _type *, const int *, const _type *, const int *,  \
      const _type *, _type *, const int *);                                   \
  void gemm(const char *transA, const char *transB, int m, int n, int k,      \
            _type alpha, const _type a[], int lda, const _type b[], int ldb,  \
            _type beta, _type c[], int ldc) {                                 \
    _system_name(transA, transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, \
                 c, &ldc);                                                    \
  }

ENABLE_SYSTEM_GEMM(float, sgemm_)
ENABLE_SYSTEM_GEMM(double, dgemm_)

#undef ENABLE_SYSTEM_GEMM

template <typename Gemm, typename T, typename Container, typename Executor>
void test(int r, int m, int n, int k, T alpha, const Container &dataA, int lda,
          const Container &dataB, int ldb, T beta, Container dataC, int ldc,
          const Container &refC, Executor &ex) {
  using etype = typename Gemm::value_type;
  std::cout << "\n=== Testing " << Gemm::get_type_string()
            << " ===" << std::endl;
  {
    auto m_a_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataA.data()), dataA.size());

    auto m_b_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataB.data()), dataB.size());

    auto m_c_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataC.data()), dataC.size());

    auto accA = make_matrix_view(ex, m_a_gpu, m, k, lda, Access::ColMajor());
    auto accB = make_matrix_view(ex, m_b_gpu, k, n, ldb, Access::ColMajor());
    auto accC = make_matrix_view(ex, m_c_gpu, m, n, ldc, Access::ColMajor());
    auto gemm = Gemm(accA, accB, accC, alpha, beta);
    run_test(r, 2.0 * m * n * k, [&] {
      auto event = ex.gemm_executor(gemm);
      ex.wait(event);
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}

template <typename T, typename Container, typename Executor>
void test_syclblas(int r, char transA, char transB, int m, int n, int k,
                   T alpha, const Container &dataA, int lda,
                   const Container &dataB, int ldb, T beta, Container dataC,
                   int ldc, const Container &refC, Executor &ex) {
  using etype = typename Container::value_type;
  std::cout << "\n=== Testing SYCL-BLAS gemm ===" << std::endl;
  {
    auto m_a_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataA.data()), dataA.size());
    auto m_b_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataB.data()), dataB.size());
    auto m_c_gpu = blas::helper::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(dataC.data()), dataC.size());
    run_test(r, 2.0 * m * n * k, [&] {
      auto event = _gemm(ex, transA, transB, m, n, k, alpha, m_a_gpu, lda,
                         m_b_gpu, ldb, beta, m_c_gpu, ldc);
      ex.wait(event);
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}

template <bool TransA, bool TransB, typename E>
void run_gemm_tests(int seed, int m, int k, int n, int rep) {
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
  run_test(rep, 2.0 * m * n * k, [&] {
    gemm(ta_str, tb_str, m, n, k, E(1), dataA.data(), lda, dataB.data(), ldb,
         E(1), refC.data(), m);
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

  Executor<SYCL> ex(q);

#define ARG m, n, k, E(1), dataA, lda, dataB, ldb, E(1), origC, ldc, refC, ex

  const int cls = 64;     // size of cache line in bytes
  const bool db = false;  // use double buffer
  const bool ba = false;  // avoid bank conflicts for A
  const bool bb = false;  // avoid bank conflicts for B
  const bool ta = TransA;
  const bool tb = TransB;
  using data_t = typename MatrixViewTypeTrace<Executor<SYCL>, E, int>::Type;

#define TARG(_tir, _tic, _twr, _twc, _ttr, _ttc) \
  GemmFactory<data_t, data_t, db, ba, bb, cls,   \
              Tile<_tir, _tic, _twr, _twc, _ttr, _ttc>, ta, tb, E>
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

#define TARG(_tir, _tic, _twr, _twc, _ttr, _ttc) \
  NoLocalGemmFactory<data_t, data_t, cls,        \
                     Tile<_tir, _tic, _twr, _twc, _ttr, _ttc>, ta, tb, E>
  test<TARG(8, 8, 8, 8, 1, 1)>(rep, ARG);
  test<TARG(8, 8, 16, 16, 1, 1)>(rep, ARG);

#undef TARG

  test_syclblas(rep, *ta_str, *tb_str, ARG);

  test<ReferenceGemmFactory<data_t, data_t, 128, ta, tb, E>>(rep, ARG);
#undef ARG
}
