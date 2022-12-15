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
 *  @filename rotg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name() {
  std::ostringstream str{};
  str << "BM_Rotg<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         bool* success) {
  // Create data
  scalar_t a = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t b = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t c = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t s = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  auto buf_a = blas::make_sycl_iterator_buffer<scalar_t>(&a, 1);
  auto buf_b = blas::make_sycl_iterator_buffer<scalar_t>(&b, 1);
  auto buf_c = blas::make_sycl_iterator_buffer<scalar_t>(&c, 1);
  auto buf_s = blas::make_sycl_iterator_buffer<scalar_t>(&s, 1);

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

  auto buf_verify_a = blas::make_sycl_iterator_buffer<scalar_t>(&a_verify, 1);
  auto buf_verify_b = blas::make_sycl_iterator_buffer<scalar_t>(&b_verify, 1);
  auto buf_verify_c = blas::make_sycl_iterator_buffer<scalar_t>(&c_verify, 1);
  auto buf_verify_s = blas::make_sycl_iterator_buffer<scalar_t>(&s_verify, 1);

  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);
  _rotg(sb_handle, buf_verify_a, buf_verify_b, buf_verify_c, buf_verify_s);

  auto event1 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_c,
                                           &c_verify, 1);
  auto event2 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_s,
                                           &s_verify, 1);
  auto event3 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_a,
                                           &a_verify, 1);
  auto event4 = blas::helper::copy_to_host(sb_handle.get_queue(), buf_verify_b,
                                           &b_verify, 1);

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
#endif

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
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
