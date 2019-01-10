/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename syclblas_benchmark.cpp
 *
 **************************************************************************/

#include "utils.hpp"

#include <interface/blas2_interface.hpp>

template <typename ScalarT>
void BM_Gemv(benchmark::State& state) {
  // Standard test setup.
  using IndexType = unsigned int;

  const char* t_str =
      "n";  // benchmark::utils::from_transpose_enum(
            // static_cast<benchmark::utils::Transposition>(state.range(0)));
  const IndexType m = static_cast<IndexType>(state.range(1));
  const IndexType n = static_cast<IndexType>(state.range(2));

  IndexType vlen = t_str[0] == 'n' ? n : m;
  IndexType rlen = t_str[0] == 'n' ? m : n;

  IndexType lda = m;
  long incX = 1;
  long incY = 1;

  state.counters["m"] = m;
  state.counters["n"] = n;

  std::cout << "t_str: " << t_str << " m: " << m << " n: " << n << std::endl;

  blas::Executor<SYCL> ex = *getExecutor();

  // Create data
  // Scalars
  ScalarT alpha = benchmark::utils::random_scalar<ScalarT>();
  ScalarT beta = benchmark::utils::random_scalar<ScalarT>();
  std::cout << "alpha: " << alpha << " beta: " << beta << std::endl;

  // Input matrix/vector, output vector.
  std::vector<ScalarT> a_m = benchmark::utils::random_data<ScalarT>(m * n);
  std::vector<ScalarT> b_v = benchmark::utils::random_data<ScalarT>(vlen);
  std::vector<ScalarT> c_v_gpu_result =
      benchmark::utils::const_data<ScalarT>(rlen, 0);

  std::cout << "Lens: a, " << a_m.size() << ", b, " << b_v.size() << ", c, "
            << c_v_gpu_result.size() << std::endl;

  //   auto m_a_gpu = blas::helper::make_sycl_iterator_buffer<ScalarT>(a_m, m *
  //   n); auto v_b_gpu = blas::helper::make_sycl_iterator_buffer<ScalarT>(b_v,
  //   vlen); auto v_c_gpu =
  //       blas::helper::make_sycl_iterator_buffer<ScalarT>(c_v_gpu_result,
  //       rlen);

  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(vlen);
  auto v_c_gpu = ex.template allocate<ScalarT>(rlen);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, vlen);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, rlen);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemv(ex, 'n', m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu, incY);
  }

  //   std::cout << "Ran successfully! " << std::endl;

  // Measure
  for (auto _ : state) {
    //     // Run
    //     auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu,
    //     incX, beta,
    //                        v_c_gpu, incY);
    //     ex.wait(event);

    //     // Report
    state.PauseTiming();
    state.counters["event_time"] = 20;
    // benchmark::utils::time_event(event);
    state.ResumeTiming();
  }
}

static void gemv_args(benchmark::internal::Benchmark* b) {
  for (int i = 2 << 5; i <= 2 << 18; i *= 2)
    for (int j = 2 << 5; j <= 2 << 18; j *= 2) {
      b->Args({(int)benchmark::utils::to_transpose_enum("n"), i, j});
      b->Args({(int)benchmark::utils::to_transpose_enum("t"), i, j});
      b->Args({(int)benchmark::utils::to_transpose_enum("c"), i, j});
    }
}

BENCHMARK_TEMPLATE(BM_Gemv, float)->Apply(gemv_args);
BENCHMARK_TEMPLATE(BM_Gemv, double)->Apply(gemv_args);

// using namespace blas;

// BENCHMARK_NAME_FORMAT(syclblas_level_2) {
//   std::ostringstream fname;
//   fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_"
//         << std::get<0>(params) << "_" << std::get<1>(params) << "_"
//         << std::get<2>(params);
//   return fname.str();
// }

// BENCHMARK(gemv, syclblas_level_2) {
//   using ScalarT = ElemT;
//   using IndexType = unsigned int;

//   const char* t_str = std::get<0>(params);
//   const IndexType m = std::get<1>(params);
//   const IndexType n = std::get<2>(params);

//   IndexType vlen = t_str[0] == 'n' ? n : m;
//   IndexType rlen = t_str[0] == 'n' ? m : n;

//   size_t n_fl_ops =
//       static_cast<size_t>(m) * static_cast<size_t>(n) *
//       static_cast<size_t>(2);

//   IndexType lda = m;
//   long incX = 1;
//   long incY = 1;

//   ScalarT alpha = benchmark<>::random_scalar<ScalarT>();
//   ScalarT beta = benchmark<>::random_scalar<ScalarT>();

//   // Input matrix
//   std::vector<ScalarT> a_m = benchmark<>::random_data<ScalarT>(m * n);
//   // Input Vector
//   std::vector<ScalarT> b_v = benchmark<>::random_data<ScalarT>(vlen);
//   // output Vector
//   std::vector<ScalarT> c_v_gpu_result =
//       benchmark<>::const_data<ScalarT>(rlen, 0);

//   auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
//   auto v_b_gpu = ex.template allocate<ScalarT>(vlen);
//   auto v_c_gpu = ex.template allocate<ScalarT>(rlen);
//   ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
//   ex.copy_to_device(b_v.data(), v_b_gpu, vlen);
//   ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, rlen);

//   benchmark<>::datapoint_t flops =
//       benchmark<>::measure(reps, n_fl_ops, [&]() -> cl::sycl::event {
//         auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu,
//         incX,
//                            beta, v_c_gpu, incY);
//         ex.wait(event);
//         return event;
//       });

//   auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), rlen);

//   ex.wait(event);

//   ex.template deallocate<ScalarT>(m_a_gpu);
//   ex.template deallocate<ScalarT>(v_b_gpu);
//   ex.template deallocate<ScalarT>(v_c_gpu);

//   return flops;
// }

// SUITE(ADD(gemv))

// SYCL_BENCHMARK_MAIN(default_ranges::level_2, 10)
