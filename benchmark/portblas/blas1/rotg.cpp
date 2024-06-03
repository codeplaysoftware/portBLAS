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
 *  @filename rotg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotg;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Create data
  scalar_t a = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t b = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t c = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t s = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  auto buf_a = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_b = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_c = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_s = blas::helper::allocate<mem_alloc, scalar_t>(1, q);

  auto copy_a = blas::helper::copy_to_device<scalar_t>(q, &a, buf_a, 1);
  auto copy_b = blas::helper::copy_to_device<scalar_t>(q, &b, buf_b, 1);
  auto copy_c = blas::helper::copy_to_device<scalar_t>(q, &c, buf_c, 1);
  auto copy_s = blas::helper::copy_to_device<scalar_t>(q, &s, buf_s, 1);

  sb_handle.wait({copy_a, copy_b, copy_c, copy_s});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t a_ref = a;
  scalar_t b_ref = b;
  scalar_t c_ref = c;
  scalar_t s_ref = s;

  scalar_t a_verify = a;
  scalar_t b_verify = b;
  scalar_t c_verify = c;
  scalar_t s_verify = s;

  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);

  auto buf_verify_a = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_verify_b = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_verify_c = blas::helper::allocate<mem_alloc, scalar_t>(1, q);
  auto buf_verify_s = blas::helper::allocate<mem_alloc, scalar_t>(1, q);

  auto copy_verify_a =
      blas::helper::copy_to_device<scalar_t>(q, &a_verify, buf_verify_a, 1);
  auto copy_verify_b =
      blas::helper::copy_to_device<scalar_t>(q, &b_verify, buf_verify_b, 1);
  auto copy_verify_c =
      blas::helper::copy_to_device<scalar_t>(q, &c_verify, buf_verify_c, 1);
  auto copy_verify_s =
      blas::helper::copy_to_device<scalar_t>(q, &s_verify, buf_verify_s, 1);

  sb_handle.wait({copy_verify_a, copy_verify_b, copy_verify_c, copy_verify_s});

  auto rotg_event =
      _rotg(sb_handle, buf_verify_a, buf_verify_b, buf_verify_c, buf_verify_s);
  sb_handle.wait(rotg_event);

  auto event1 = blas::helper::copy_to_host(q, buf_verify_c, &c_verify, 1);
  auto event2 = blas::helper::copy_to_host(q, buf_verify_s, &s_verify, 1);
  auto event3 = blas::helper::copy_to_host(q, buf_verify_a, &a_verify, 1);
  auto event4 = blas::helper::copy_to_host(q, buf_verify_b, &b_verify, 1);

  sb_handle.wait({event1, event2, event3, event4});

  const bool isAlmostEqual =
      utils::almost_equal<scalar_t, scalar_t>(a_verify, a_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(b_verify, b_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(c_verify, c_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(s_verify, s_ref);

  if (!isAlmostEqual) {
    std::ostringstream err_stream;
    err_stream << "Value mismatch: " << std::endl;
    err_stream << "Got: "
               << "A: " << a_verify << " B: " << b_verify << " C: " << c_verify
               << " S: " << s_verify << std::endl;
    err_stream << "Expected: "
               << "A: " << a_ref << " B: " << b_ref << " C: " << c_ref
               << " S: " << s_ref << std::endl;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };

  blas::helper::deallocate<mem_alloc>(buf_verify_a, q);
  blas::helper::deallocate<mem_alloc>(buf_verify_b, q);
  blas::helper::deallocate<mem_alloc>(buf_verify_c, q);
  blas::helper::deallocate<mem_alloc>(buf_verify_s, q);
#endif

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<sycl::event> {
    auto event = _rotg(sb_handle, buf_a, buf_b, buf_c, buf_s);
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

  blas::helper::deallocate<mem_alloc>(buf_a, q);
  blas::helper::deallocate<mem_alloc>(buf_b, q);
  blas::helper::deallocate<mem_alloc>(buf_c, q);
  blas::helper::deallocate<mem_alloc>(buf_s, q);
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
