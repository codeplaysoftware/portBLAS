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
void run(benchmark::State& state, ExecutorType* executorPtr, bool* success) {
  // Create data
  scalar_t a = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t b = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t c = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t s = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  ExecutorType& ex = *executorPtr;

  auto buf_a = utils::make_quantized_buffer<scalar_t>(ex, a);
  auto buf_b = utils::make_quantized_buffer<scalar_t>(ex, b);
  auto buf_c = utils::make_quantized_buffer<scalar_t>(ex, c);
  auto buf_s = utils::make_quantized_buffer<scalar_t>(ex, s);
  
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

  auto buf_verify_a = utils::make_quantized_buffer<scalar_t>(ex, a_verify);
  auto buf_verify_b = utils::make_quantized_buffer<scalar_t>(ex, b_verify);
  auto buf_verify_c = utils::make_quantized_buffer<scalar_t>(ex, c_verify);
  auto buf_verify_s = utils::make_quantized_buffer<scalar_t>(ex, s_verify);

  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);
  _rotg(ex, buf_verify_a, buf_verify_b, buf_verify_c, buf_verify_s);

  auto event3 =
      ex.get_policy_handler().copy_to_host(buf_verify_a, &a_verify, 1);
  auto event4 =
      ex.get_policy_handler().copy_to_host(buf_verify_b, &b_verify, 1);
  auto event1 =
      ex.get_policy_handler().copy_to_host(buf_verify_c, &c_verify, 1);
  auto event2 =
      ex.get_policy_handler().copy_to_host(buf_verify_s, &s_verify, 1);
  ex.get_policy_handler().wait(event1);
  ex.get_policy_handler().wait(event2);
  ex.get_policy_handler().wait(event3);
  ex.get_policy_handler().wait(event4);

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
    auto event = _rotg(ex, buf_a, buf_b, buf_c, buf_s);
    ex.get_policy_handler().wait(event);
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
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas1_params(args);

  auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                       bool* success) { run<scalar_t>(st, exPtr, success); };
  benchmark::RegisterBenchmark(get_name<scalar_t>().c_str(), BM_lambda, exPtr,
                               success);
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
