/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2022 Codeplay Software Limited
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
 *  @filename rotmg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name() {
  std::ostringstream str{};
  str << "BM_Rotmg<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/";
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         bool* success) {
  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param = std::vector<scalar_t>(param_size);
  scalar_t d1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t d2 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t x1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t y1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  auto buf_d1 = blas::make_sycl_iterator_buffer<scalar_t>(&d1, 1);
  auto buf_d2 = blas::make_sycl_iterator_buffer<scalar_t>(&d2, 1);
  auto buf_x1 = blas::make_sycl_iterator_buffer<scalar_t>(&x1, 1);
  auto buf_y1 = blas::make_sycl_iterator_buffer<scalar_t>(&y1, 1);
  auto buf_param = blas::make_sycl_iterator_buffer<scalar_t>(param, param_size);

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

  auto buf_verify_d1 = blas::make_sycl_iterator_buffer<scalar_t>(&d1_verify, 1);
  auto buf_verify_d2 = blas::make_sycl_iterator_buffer<scalar_t>(&d2_verify, 1);
  auto buf_verify_x1 = blas::make_sycl_iterator_buffer<scalar_t>(&x1_verify, 1);
  auto buf_verify_y1 = blas::make_sycl_iterator_buffer<scalar_t>(&y1_verify, 1);
  auto device_param =
      blas::make_sycl_iterator_buffer<scalar_t>(param_verify, param_size);

  reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());
  _rotmg(sb_handle, buf_verify_d1, buf_verify_d2, buf_verify_x1, buf_verify_y1,
         device_param);

  auto event1 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_d1,
                                           &d1_verify, 1);
  auto event2 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_d2,
                                           &d2_verify, 1);
  auto event3 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_x1,
                                           &x1_verify, 1);
  auto event4 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_y1,
                                           &y1_verify, 1);
  auto event5 = blas::helper::copy_to_host(sb_handle.get_queue(), device_param,
                                           param_verify.data(), param_size);

  sb_handle.wait({event1, event2, event3, event4, event5});

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
  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
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

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                       bool* success) {
    run<scalar_t>(st, sb_handle_ptr, success);
  };
  benchmark::RegisterBenchmark(get_name<scalar_t>().c_str(), BM_lambda,
                               sb_handle_ptr, success);
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
