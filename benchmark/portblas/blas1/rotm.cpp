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
 *  @filename rotm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotm;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

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

  auto gpu_x_v = blas::helper::allocate<mem_alloc, scalar_t>(size, q);
  auto gpu_y_v = blas::helper::allocate<mem_alloc, scalar_t>(size, q);
  auto gpu_param = blas::helper::allocate<mem_alloc, scalar_t>(param_size, q);

  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, x_v.data(), gpu_x_v, size);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, y_v.data(), gpu_y_v, size);
  auto copy_param = blas::helper::copy_to_device<scalar_t>(
      q, param.data(), gpu_param, param_size);

  sb_handle.wait({copy_x, copy_y, copy_param});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_v_ref = x_v;
  std::vector<scalar_t> y_v_ref = y_v;

  std::vector<scalar_t> x_v_verify = x_v;
  std::vector<scalar_t> y_v_verify = y_v;

  reference_blas::rotm(size, x_v_ref.data(), 1, y_v_ref.data(), 1,
                       param.data());
  {
    auto gpu_x_verify = blas::helper::allocate<mem_alloc, scalar_t>(size, q);
    auto gpu_y_verify = blas::helper::allocate<mem_alloc, scalar_t>(size, q);

    auto copy_x_verify = blas::helper::copy_to_device<scalar_t>(
        q, x_v_verify.data(), gpu_x_verify, size);
    auto copy_y_verify = blas::helper::copy_to_device<scalar_t>(
        q, y_v_verify.data(), gpu_y_verify, size);

    sb_handle.wait({copy_x_verify, copy_y_verify});

    auto rotm_event =
        _rotm(sb_handle, size, gpu_x_verify, static_cast<index_t>(1),
              gpu_y_verify, static_cast<index_t>(1), gpu_param);
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

  auto blas_method_def = [&]() -> std::vector<sycl::event> {
    auto event = _rotm(sb_handle, size, gpu_x_v, static_cast<index_t>(1),
                       gpu_y_v, static_cast<index_t>(1), gpu_param);
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

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<blas1_param_t> params) {
  for (auto size : params) {
    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         index_t size, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, size, success);
    };

    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            size, mem_type).c_str(),
        BM_lambda, sb_handle_ptr, size, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto rotm_params = blas_benchmark::utils::get_blas1_params(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER, rotm_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, rotm_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK_FLOAT(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
