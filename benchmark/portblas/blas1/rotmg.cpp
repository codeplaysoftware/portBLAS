/***************************************************************************
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
 *  @filename rotmg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotmg;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::rotmg, scalar_t>(state, 1);
  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param = std::vector<scalar_t>(param_size);
  scalar_t d1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t d2 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t x1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t y1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  auto buf_d1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_d2 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_x1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_y1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_param = blas::helper::allocate<mem_alloc, scalar_t>(param_size, q);

  auto copy_d1 = blas::helper::copy_to_device<scalar_t>(q, &d1, buf_d1, 1);
  auto copy_d2 = blas::helper::copy_to_device<scalar_t>(q, &d2, buf_d2, 1);
  auto copy_x1 = blas::helper::copy_to_device<scalar_t>(q, &x1, buf_x1, 1);
  auto copy_y1 = blas::helper::copy_to_device<scalar_t>(q, &y1, buf_y1, 1);
  auto copy_param = blas::helper::copy_to_device<scalar_t>(
      q, param.data(), buf_param, param_size);

  sb_handle.wait({copy_d1, copy_d2, copy_x1, copy_y1, copy_param});
#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t d1_ref = d1;
  scalar_t d2_ref = d2;
  scalar_t x1_ref = x1;
  scalar_t y1_ref = y1;
  std::vector<scalar_t> param_ref = param;

  scalar_t d1_verify = d1;
  scalar_t d2_verify = d2;
  scalar_t x1_verify = x1;
  scalar_t y1_verify = y1;
  std::vector<scalar_t> param_verify = param;

  reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());
  {
    auto buf_verify_d1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
    auto buf_verify_d2 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
    auto buf_verify_x1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
    auto buf_verify_y1 = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
    auto buf_verify_param =
        blas::helper::allocate<mem_alloc, scalar_t>(param_size, q);

    auto copy_verify_d1 =
        blas::helper::copy_to_device<scalar_t>(q, &d1_verify, buf_verify_d1, 1);
    auto copy_verify_d2 =
        blas::helper::copy_to_device<scalar_t>(q, &d2_verify, buf_verify_d2, 1);
    auto copy_verify_x1 =
        blas::helper::copy_to_device<scalar_t>(q, &x1_verify, buf_verify_x1, 1);
    auto copy_verify_y1 =
        blas::helper::copy_to_device<scalar_t>(q, &y1_verify, buf_verify_y1, 1);
    auto copy_verify_param = blas::helper::copy_to_device<scalar_t>(
        q, param_verify.data(), buf_verify_param, param_size);

    sb_handle.wait({copy_verify_d1, copy_verify_d2, copy_verify_x1,
                    copy_verify_y1, copy_verify_param});

    auto rotmg_event = _rotmg(sb_handle, buf_verify_d1, buf_verify_d2,
                              buf_verify_x1, buf_verify_y1, buf_verify_param);
    sb_handle.wait(rotmg_event);

    auto event1 = blas::helper::copy_to_host(q, buf_verify_d1, &d1_verify, 1);
    auto event2 = blas::helper::copy_to_host(q, buf_verify_d2, &d2_verify, 1);
    auto event3 = blas::helper::copy_to_host(q, buf_verify_x1, &x1_verify, 1);
    auto event4 = blas::helper::copy_to_host(q, buf_verify_y1, &y1_verify, 1);
    auto event5 = blas::helper::copy_to_host(q, buf_verify_param,
                                             param_verify.data(), param_size);

    sb_handle.wait({event1, event2, event3, event4, event5});

    blas::helper::deallocate<mem_alloc>(buf_verify_d1, q);
    blas::helper::deallocate<mem_alloc>(buf_verify_d2, q);
    blas::helper::deallocate<mem_alloc>(buf_verify_x1, q);
    blas::helper::deallocate<mem_alloc>(buf_verify_y1, q);
    blas::helper::deallocate<mem_alloc>(buf_verify_param, q);
  }

  const bool isAlmostEqual = utils::almost_equal(d1_verify, d1_ref) &&
                             utils::almost_equal(d2_verify, d2_ref) &&
                             utils::almost_equal(x1_verify, x1_ref) &&
                             utils::almost_equal(y1_verify, y1_ref);

  if (!isAlmostEqual) {
    std::ostringstream err_stream;
    err_stream << "Value mismatch." << std::endl;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<sycl::event> {
    auto event = _rotmg(sb_handle, buf_d1, buf_d2, buf_x1, buf_y1, buf_param);
    sb_handle.wait(event);
    return event;
  };

  // Warm up to avoid benchmarking data transfer
  blas_benchmark::utils::warmup(blas_method_def);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(buf_d1, q);
  blas::helper::deallocate<mem_alloc>(buf_d2, q);
  blas::helper::deallocate<mem_alloc>(buf_x1, q);
  blas::helper::deallocate<mem_alloc>(buf_y1, q);
  blas::helper::deallocate<mem_alloc>(buf_param, q);
};

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type) {
  auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                       bool* success) {
    run<scalar_t, mem_alloc>(st, sb_handle_ptr, success);
  };
  benchmark::RegisterBenchmark(
      blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
          mem_type).c_str(),
      BM_lambda, sb_handle_ptr, success)
      ->UseRealTime();
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(sb_handle_ptr,
                                                             success, blas_benchmark::utils::MEM_TYPE_USM);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
