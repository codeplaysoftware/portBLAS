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
 *  @filename gemm_test.hpp
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <tuple>
#include <vector>


#include <CL/sycl.hpp>


#include <operations/blas3_trees_gemm.hpp>
#include <utils/vec.hpp>


#include "test_utils.hpp"


using namespace cl::sycl;
// using namespace blas;


template <typename T>
void reference_gemm(int m, int n, int k, T *A, int lda, T *B, int ldb, T *C,
                    int ldc) {
  for(int j = 0; j < n; ++j) {
    for(int p = 0; p < k; ++p) {
      for(int i = 0; i < m; ++i) {
        C[i + j*ldc] += A[i + p*lda] * B[p + j*ldb];
      }
    }
  }
}


template <int, typename, typename> class VectorGemvTest;
template <int vec_size, typename T, typename RndEngine>
void vector_gemv_test(queue q, RndEngine rnd) {
  const int fst_dim = vec_size;
  const int snd_dim = 1;
  const int int_dim = vec_size;
  std::vector<T> A = gen_matrix(fst_dim, int_dim, T(-1), T(1), rnd);
  std::vector<T> B = gen_matrix(int_dim, snd_dim, T(-1), T(1), rnd);
  std::vector<T> C = gen_matrix(fst_dim, snd_dim, T(-1), T(1), rnd);
  std::vector<T> refC = C;

  reference_gemm(fst_dim, snd_dim, int_dim, A.data(), fst_dim, B.data(),
                 int_dim, refC.data(), fst_dim);
  {
    buffer<T, 1> bA(A.data(), range<1>(fst_dim*int_dim));
    buffer<T, 1> bB(B.data(), range<1>(int_dim*snd_dim));
    buffer<T, 1> bC(C.data(), range<1>(fst_dim*snd_dim));
    q.submit([&](handler &cgh) {
      auto aA = bA.template get_access<access::mode::read>(cgh);
      auto aB = bB.template get_access<access::mode::read>(cgh);
      auto aC = bC.template get_access<access::mode::read_write>(cgh);
      cgh.single_task<class VectorGemvTest<vec_size, T, RndEngine>>([=]() {
        blas::vec<T, vec_size> pA[vec_size];
        blas::vec<T, vec_size> pB;
        blas::vec<T, vec_size> pC;
        #pragma unroll
        for (int i = 0; i < vec_size; ++i) {
          pA[i].load(i, aA);
        }
        pB.load(0, aB);
        pC.load(0, aC);
        blas::vector::_gemv<vec_size>(pA, pB, pC);
        pC.store(0, aC);
      });
    });
  }
  std::cout << "Err: " << std::scientific
            << relative_diff(refC, C) << std::endl;
};


// compile-time for loop over vector sizes
template <typename T, typename... Args>
inline void vector_gemv_test_all(static_list<>, Args...) {}


template <typename T, int val, int... rest, typename... Args>
inline void vector_gemv_test_all(static_list<val, rest...>, Args... args) {
  std::cout << "vector_gemv_test<" << val << ", "
            << type_name<T>::name << ">" << std::endl;
  vector_gemv_test<val, T>(args...);
  vector_gemv_test_all<T>(static_list<rest...>(), args...);
}
// end compile-time for loop


template <int, int, int, typename, typename> class ThreadGemvTest;
template <int scratch_width, int thr_size, int vec_size, typename T,
          typename Rnd>
void thread_gemm_test(queue q, Rnd rnd) {
  const int fst_dim = vec_size;
  const int snd_dim = scratch_width;
  const int int_dim = thr_size*vec_size;
  std::vector<T> A = gen_matrix(fst_dim, int_dim, T(-1), T(1), rnd);
  std::vector<T> B = gen_matrix(int_dim, snd_dim, T(-1), T(1), rnd);
  std::vector<T> C = gen_matrix(fst_dim, snd_dim, T(-1), T(1), rnd);
  std::vector<T> refC = C;

  reference_gemm(fst_dim, snd_dim, int_dim, A.data(), fst_dim, B.data(),
                 int_dim, refC.data(), fst_dim);
  {
    buffer<T, 1> bA(A.data(), range<1>(fst_dim*int_dim));
    buffer<T, 1> bB(B.data(), range<1>(int_dim*snd_dim));
    buffer<T, 1> bC(C.data(), range<1>(fst_dim*snd_dim));
    q.submit([&](handler &cgh) {
      auto aA = bA.template get_access<access::mode::read>(cgh);
      auto aB = bB.template get_access<access::mode::read>(cgh);
      auto aC = bC.template get_access<access::mode::read_write>(cgh);
      cgh.single_task
            <class ThreadGemvTest<scratch_width, thr_size, vec_size, T, Rnd>>
            ([=]() {
        blas::vec<T, vec_size> pA[vec_size*thr_size];
        blas::vec<T, vec_size> pC[scratch_width];
        #pragma unroll
        for (int i = 0; i < vec_size*thr_size; ++i) {
          pA[i].load(i, aA);
        }
        #pragma unroll
        for (int i = 0; i < scratch_width; ++i) {
          pC[i].load(i, aC);
        }
        blas::thread::_gemm<scratch_width, thr_size, vec_size>(pA, aB, pC);
        #pragma unroll
        for (int i = 0; i < scratch_width; ++i) {
          pC[i].store(i, aC);
        }
      });
    });
  }
  std::cout << "Err: " << std::scientific
            << relative_diff(refC, C) << std::endl;
};


// compile-time nested for loop over scratch_width, thr and vec sizes
template <int val1, int val2, typename T, typename... Args>
inline void thread_gemm_test_all(static_list<>, Args...) {}


template <int val1, int val2, typename T, int val3, int... rest,
          typename... Args>
inline void thread_gemm_test_all(static_list<val3, rest...>, Args... args) {
  std::cout << "thread_gemm_test<" << val1 << ", " << val2 << ", "
            << val3 << ", " << type_name<T>::name << ">" << std::endl;
  thread_gemm_test<val1, val2, val3, T>(args...);
  thread_gemm_test_all<val1, val2, T>(static_list<rest...>(), args...);
}


template <int val1, typename T, int... inner, typename... Args>
inline void thread_gemm_test_all(
    static_list<>, static_list<inner...>, Args...) {}


template <int val1, typename T, int val2, int... rest, int... inner,
          typename... Args>
inline void thread_gemm_test_all(
    static_list<val2, rest...>, static_list<inner...> inner_list,
    Args... args) {
  thread_gemm_test_all<val1, val2, T>(inner_list, args...);
  thread_gemm_test_all<val1, T>(static_list<rest...>(), inner_list, args...);
}


template <typename T, int... middle, int... inner, typename... Args>
inline void thread_gemm_test_all(
    static_list<>, static_list<middle...>, static_list<inner...>, Args...) {}


template <typename T, int val1, int... rest, int... middle, int... inner,
          typename... Args>
inline void thread_gemm_test_all(
    static_list<val1, rest...>, static_list<middle...> middle_list,
    static_list<inner...> inner_list, Args... args) {
  thread_gemm_test_all<val1, T>(middle_list, inner_list, args...);
  thread_gemm_test_all<T>(static_list<rest...>(), middle_list, inner_list,
                          args...);
}

// end compile-time nested for loop


int main() {
  const int seed = 42;
  queue q;
  std::mt19937 rnd(seed);

  std::cout << "=== Testing vector gemv ===" << std::endl;
  const static_list<1, 2, 4, 8, 16> vec_size_list{};
  vector_gemv_test_all<float>(vec_size_list, q, rnd);
  vector_gemv_test_all<double>(vec_size_list, q, rnd);

  std::cout << "\n=== Testing thread gemm ===" << std::endl;
  const static_list<8, 16> scratch_width_list{};
  const static_list<1, 2, 4> thr_size_list{};
  const static_list<1, 2, 8> reduced_vec_size_list{};
  thread_gemm_test_all<float>(
      scratch_width_list, thr_size_list, reduced_vec_size_list, q, rnd);
  thread_gemm_test_all<double>(
      scratch_width_list, thr_size_list, reduced_vec_size_list, q, rnd);

  return 0;
}

