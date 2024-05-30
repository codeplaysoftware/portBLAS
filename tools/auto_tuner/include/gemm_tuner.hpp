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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename gemm_tuner.hpp
 *
 **************************************************************************/

#include "tune.hpp"
#include "tuner_types.hpp"
#include "utils.hpp"

#include "reference_gemm.hpp"
#include "portblas.hpp"

using namespace cl::sycl;
using namespace blas;
// Convert batch_type=strided to interleaved on the host
template <typename scalar_t>
inline std::vector<scalar_t> strided_to_interleaved(
    const std::vector<scalar_t> &input, int offset, int ld_rows, int ld_cols,
    int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int c = 0; c < ld_cols; ++c) {
    for (int r = 0; r < ld_rows; ++r) {
      for (int b = 0; b < batchs; ++b) {
        output[c * ld_rows * batchs + r * batchs + b + offset] =
            input[b * ld_cols * ld_rows + c * ld_rows + r + offset];
      }
    }
  }
  return output;
}

// Convert batch_type=interleaved to strided on the host
template <typename scalar_t>
inline std::vector<scalar_t> interleaved_to_strided(
    const std::vector<scalar_t> &input, int offset, int ld_rows, int ld_cols,
    int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int b = 0; b < batchs; ++b) {
    for (int c = 0; c < ld_cols; ++c) {
      for (int r = 0; r < ld_rows; ++r) {
        output[b * ld_cols * ld_rows + c * ld_rows + r + offset] =
            input[c * ld_rows * batchs + r * batchs + b + offset];
      }
    }
  }
  return output;
}

template <typename T>
static TestResultEntry tune_portblas(portblas_handle_t &sb_handle, int r,
                                     char transA, char transB, GemmArgs<T> a,
                                     ::blas::gemm_batch_type_t batch_type) {
  TestResultEntry result("portBLAS gemm");
  {
    auto event = blas::helper::copy_to_device(
        sb_handle.get_queue(), a.init_c.data(), a.c, a.init_c.size());
    event.wait_and_throw();

    const double flop_count = 2.0 * a.m * a.n * a.k * a.batch_size;
    run_tune(r, flop_count, result, [&] {
      auto event_list = _gemm_batched(sb_handle, transA, transB, a.m, a.n, a.k,
                                      a.alpha, a.a, a.lda, a.b, a.ldb, a.beta,
                                      a.c, a.ldc, a.batch_size, batch_type);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    });
  }
  {
    auto event = blas::helper::copy_to_host(
        sb_handle.get_queue(), a.c, a.output_c.data(), a.output_c.size());
    event.wait_and_throw();
  }

  result.error = relative_diff(a.expected_c, a.output_c);
  return result;
}

template <bool TransA, bool TransB, typename DataType>
void run_tune_gemm(portblas_handle_t &sb_handle, int seed, int m, int k, int n,
                   int batch_size, int rep,
                   ::blas::gemm_batch_type_t batch_type) {
  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto host_a = get_random_vector<DataType>(k * m * batch_size, -1, 1, rnd);
  auto host_b = get_random_vector<DataType>(n * k * batch_size, -1, 1, rnd);
  auto host_c = get_random_vector<DataType>(m * n * batch_size, -1, 1, rnd);
  auto expected_c = host_c;
  auto result_c = host_c;

  const char *ta_str = TransA ? "T" : "N";
  const char *tb_str = TransB ? "T" : "N";

  const int lda = TransA ? k : m;
  const int ldb = TransB ? n : k;
  const int ldc = m;

  const DataType alpha = 1;
  const DataType beta = 1;
  const double flop_count = 2.0 * m * n * k * batch_size;

  TestResultEntry ref_result("System GEMM implementation");
  run_tune(rep, flop_count, ref_result, [&] {
    if (batch_size > 1 && batch_type == gemm_batch_type_t::interleaved) {
      host_a =
          interleaved_to_strided(host_a, 0, lda, TransA ? m : k, batch_size);
      host_b =
          interleaved_to_strided(host_b, 0, ldb, TransB ? k : n, batch_size);
      expected_c = interleaved_to_strided(expected_c, 0, ldc, n, batch_size);
    }
    for (int bs = 0; bs < batch_size; bs++) {
      // system gemm implementation
      reference_gemm::gemm(ta_str, tb_str, m, n, k, alpha,
                           host_a.data() + (bs * m * k), lda,
                           host_b.data() + (bs * n * k), ldb, beta,
                           expected_c.data() + (bs * m * n), m);
    }
    if (batch_size > 1 && batch_type == gemm_batch_type_t::interleaved) {
      expected_c = strided_to_interleaved(expected_c, 0, ldc, n, batch_size);
      host_a =
          strided_to_interleaved(host_a, 0, lda, TransA ? m : k, batch_size);
      host_b =
          strided_to_interleaved(host_b, 0, ldb, TransB ? k : n, batch_size);
    }
  });
  ref_result.error = 0.0;

  TestResult results{};
  results.push_back(ref_result);
  const auto device_a = blas::make_sycl_iterator_buffer(host_a, host_a.size());
  const auto device_b = blas::make_sycl_iterator_buffer(host_b, host_b.size());
  auto device_c = blas::make_sycl_iterator_buffer(host_c, host_c.size());
  GemmArgs<DataType> args{m,        n,        k,   alpha,      device_a,
                          lda,      device_b, ldb, beta,       host_c,
                          device_c, result_c, ldc, batch_size, expected_c};

  {
    auto result =
        tune_portblas(sb_handle, rep, *ta_str, *tb_str, args, batch_type);
    results.push_back(result);
  }

#define BENCH_PARAMS(MEM, ALG, BATCH, VEC, ...)                             \
  do {                                                                      \
    auto result =                                                           \
        tune<__VA_ARGS__, GemmConfig<TransA, TransB, MEM, ALG, BATCH, VEC>, \
             DataType>(sb_handle, rep, args);                               \
    results.push_back(result);                                              \
  } while (0);

#include "generated_combinations.def"

#undef BENCH_PARAMS
  std::cout << "SIZE : " << results.size() << std::endl;
  sb_handle.wait();
  std::sort(results.begin(), results.end());
  results.print_all();
}
