/**************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
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
 *  @filename omatcopy.cpp
 *
 **************************************************************************/

#include "../../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n, scalar_t alpha,
                     index_t lda_mul, index_t ldb_mul) {
  std::ostringstream str{};
  str << "BM_omatcopy<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << t << "/" << m << "/" << n << "/" << alpha << "/" << lda_mul
      << "/" << ldb_mul;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_geam_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_sgeam(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dgeam(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, int ti, index_t m,
         index_t n, scalar_t alpha, index_t lda_mul, index_t ldb_mul,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  const auto lda = (*t_str == 't') ? lda_mul * n : lda_mul * m;
  const auto ldb = ldb_mul * m;

  const auto size_a = lda * ((*t_str == 't') ? m : n);
  const auto size_b = ldb * n;

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOP::omatcopy, scalar_t>(
      state, t_str, m, n, lda_mul, ldb_mul);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(size_a);
  std::vector<scalar_t> m_b =
      blas_benchmark::utils::random_data<scalar_t>(size_b);

  blas_benchmark::utils::HIPVector<scalar_t> m_a_gpu(size_a, m_a.data());
  blas_benchmark::utils::HIPVector<scalar_t> m_b_gpu(size_b, m_b.data());

  // Matrix options (rocBLAS)
  const rocblas_operation trans_rb =
      t_str[0] == 'n' ? rocblas_operation_none : rocblas_operation_transpose;

  // Dummy Variables set to only compute C:=alpha*op(A) in rocBLAS geam
  const scalar_t beta_null = 0;
  const index_t ld_null = m;
  const rocblas_operation trans_null_rb = rocblas_operation_none;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_b_ref = m_b;

  reference_blas::ext_omatcopy<false>(*t_str, m, n, alpha, m_a, lda, m_b_ref,
                                      ldb);

  std::vector<scalar_t> m_b_temp = m_b;
  {
    blas_benchmark::utils::HIPVector<scalar_t, true> m_b_temp_gpu(
        size_b, m_b_temp.data());

    rocblas_geam_f<scalar_t>(rb_handle, trans_rb, trans_null_rb, m, n, &alpha,
                             m_a_gpu, lda, &beta_null, nullptr, ld_null,
                             m_b_temp_gpu, ldb);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_b_temp, m_b_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    rocblas_geam_f<scalar_t>(rb_handle, trans_rb, trans_null_rb, m, n, &alpha,
                             m_a_gpu, lda, &beta_null, nullptr, ld_null,
                             m_b_gpu, ldb);
    return;
  };

  hipEvent_t start, stop;
  CHECK_HIP_ERROR(hipEventCreate(&start));
  CHECK_HIP_ERROR(hipEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
    CHECK_HIP_ERROR(hipEventRecord(start, NULL));
    rocblas_geam_f<scalar_t>(rb_handle, trans_rb, trans_null_rb, m, n, &alpha,
                             m_a_gpu, lda, &beta_null, nullptr, ld_null,
                             m_b_gpu, ldb);
    CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup);
  CHECK_HIP_ERROR(hipStreamSynchronize(NULL));

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef_hip(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  CHECK_HIP_ERROR(hipEventDestroy(start));
  CHECK_HIP_ERROR(hipEventDestroy(stop));
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto omatcopy_params =
      blas_benchmark::utils::get_matcopy_params<scalar_t>(args);

  for (auto p : omatcopy_params) {
    std::string ts;
    index_t m, n, lda_mul, ldb_mul;
    scalar_t alpha;
    std::tie(ts, m, n, alpha, lda_mul, ldb_mul) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle_, int t,
                         index_t m, index_t n, scalar_t alpha, index_t lda_mul,
                         index_t ldb_mul, bool* success) {
      run<scalar_t>(st, rb_handle_, t, m, n, alpha, lda_mul, ldb_mul, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts, m, n, alpha, lda_mul, ldb_mul).c_str(),
        BM_lambda, rb_handle, t, m, n, alpha, lda_mul, ldb_mul, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
