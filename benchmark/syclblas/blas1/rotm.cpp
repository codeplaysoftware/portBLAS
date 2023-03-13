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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename rotm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size, std::string mem_type) {
  std::ostringstream str{};
  str << "BM_Rotm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << mem_type << "/" << size;
  return str.str();
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::rotm, scalar_t>(state, size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param =
      blas_benchmark::utils::random_data<scalar_t>(param_size);
  param[0] =
      static_cast<scalar_t>(-1.0);  // Use -1.0 flag to use the whole matrix

  std::vector<scalar_t> x_v =
      blas_benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> y_v =
      blas_benchmark::utils::random_data<scalar_t>(size);

  typename blas::helper::AllocHelper<scalar_t, mem_alloc>::type gpu_x_v,
      gpu_y_v, gpu_param;
  cl::sycl::event copy_x, copy_y, copy_param;

  std::tie(gpu_x_v, copy_x) =
      blas::helper::allocate<mem_alloc, scalar_t>(x_v.data(), size, q);
  std::tie(gpu_y_v, copy_y) =
      blas::helper::allocate<mem_alloc, scalar_t>(y_v.data(), size, q);
  std::tie(gpu_param, copy_param) =
      blas::helper::allocate<mem_alloc, scalar_t>(param.data(), param_size, q);

  sb_handle.wait(copy_param);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_v_ref = x_v;
  std::vector<scalar_t> y_v_ref = y_v;

  std::vector<scalar_t> x_v_verify = x_v;
  std::vector<scalar_t> y_v_verify = y_v;

  reference_blas::rotm(size, x_v_ref.data(), 1, y_v_ref.data(), 1,
                       param.data());
  {
    typename blas::helper::AllocHelper<scalar_t, mem_alloc>::type gpu_x_verify,
        gpu_y_verify;
    cl::sycl::event copy_x_verify, copy_y_verify;

    std::tie(gpu_x_verify, copy_x_verify) =
        blas::helper::allocate<mem_alloc, scalar_t>(x_v_verify.data(), size, q);
    std::tie(gpu_y_verify, copy_y_verify) =
        blas::helper::allocate<mem_alloc, scalar_t>(y_v_verify.data(), size, q);

    auto rotm_event = _rotm(
        sb_handle, size, gpu_x_verify, static_cast<index_t>(1), gpu_y_verify,
        static_cast<index_t>(1), gpu_param, {copy_x_verify, copy_y_verify});
    sb_handle.wait(rotm_event);

    auto event1 =
        blas::helper::copy_to_host(q, gpu_x_verify, x_v_verify.data(), size);
    auto event2 =
        blas::helper::copy_to_host(q, gpu_y_verify, y_v_verify.data(), size);
    sb_handle.wait({event1, event2});

    blas::helper::deallocate<mem_alloc>(gpu_x_verify, q);
    blas::helper::deallocate<mem_alloc>(gpu_y_verify, q);
  }
  // Verify results
  std::ostringstream err_stream;
  const bool isAlmostEqual = utils::compare_vectors<scalar_t, scalar_t>(
                                 x_v_ref, x_v_verify, err_stream, "") &&
                             utils::compare_vectors<scalar_t, scalar_t>(
                                 y_v_ref, y_v_verify, err_stream, "");

  if (!isAlmostEqual) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event =
        _rotm(sb_handle, size, gpu_x_v, static_cast<index_t>(1), gpu_y_v,
              static_cast<index_t>(1), gpu_param, {copy_x, copy_y});
    sb_handle.wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  sb_handle.wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(gpu_x_v, q);
  blas::helper::deallocate<mem_alloc>(gpu_y_v, q);
  blas::helper::deallocate<mem_alloc>(gpu_param, q);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto rotm_params = blas_benchmark::utils::get_blas1_params(args);

#define REGISTER_MEM_TYPE_BENCHMARKS(MEMORY, NAME)                             \
  for (auto size : rotm_params) {                                              \
    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr, \
                         index_t size, bool* success) {                        \
      run<scalar_t, MEMORY>(st, sb_handle_ptr, size, success);                 \
    };                                                                         \
    benchmark::RegisterBenchmark(get_name<scalar_t>(size, NAME).c_str(),       \
                                 BM_lambda, sb_handle_ptr, size, success)      \
        ->UseRealTime();                                                       \
  }

  REGISTER_MEM_TYPE_BENCHMARKS(blas::helper::AllocType::usm, "usm");
  REGISTER_MEM_TYPE_BENCHMARKS(blas::helper::AllocType::buffer, "buffer");

#undef REGISTER_MEM_TYPE_BENCHMARKS
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK_FLOAT(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
