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
 *  @filename test_utils.hpp
 *
 **************************************************************************/

#include <cmath>
#include <functional>
#include <numeric>
#include <random>


template <typename T, typename RndEngine>
std::vector<T> gen_matrix(int m, int n, T lo, T hi, RndEngine rnd) {
  std::uniform_real_distribution<T> dst(lo, hi);
  std::vector<T> v(m*n);
  for (auto &e : v) {
    e = dst(rnd);
  }
  return v;
}


template <typename T>
T relative_diff(const std::vector<T> &ref, const std::vector<T> &obt) {
  T mag(0);
  for (auto x : ref) {
    mag += x*x;
  }
  T diff = std::inner_product(
      std::begin(ref), std::end(ref), std::begin(obt), T(0), std::plus<T>(),
      [](T x, T y) { return (x-y)*(x-y); });
  return std::sqrt(diff / mag);
}


template <typename T>
struct type_name {
  constexpr static const char * const name = "unknown";
};

#define ENABLE_TYPE_NAME(_type) \
template <> \
struct type_name<_type> { \
  constexpr static const char * const name = #_type; \
};

ENABLE_TYPE_NAME(int)
ENABLE_TYPE_NAME(float)
ENABLE_TYPE_NAME(double)


template <int...> struct static_list {};


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
              << "perf = " << flop_cnt / sec / 1e9 << " GFLOPS"
              << std::endl;
}

