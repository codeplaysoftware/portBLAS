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
 *  @filename asum.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Asum<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2.0 * size_d;
  state.counters["bytes_processed"] = size_d * sizeof(scalar_t);

  ExecutorType& ex = *executorPtr;

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  scalar_t vr;

  auto inx = blas::make_sycl_iterator_buffer<scalar_t>(v1, size);
  auto inr = blas::make_sycl_iterator_buffer<scalar_t>(&vr, 1);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t vr_ref = reference_blas::asum(size, v1.data(), 1);
  scalar_t vr_temp = 0;
  {
    auto vr_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(&vr_temp, 1);
    auto event = _asum(ex, size, inx, 1, vr_temp_gpu);
    ex.get_policy_handler().wait(event);
  }

  if (!utils::almost_equal<scalar_t>(vr_temp, vr_ref)) {
    std::cerr << "Value mismatch: " << vr_temp << "; expected " << vr_ref
              << std::endl;
    exit(1);
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _asum(ex, size, inx, 1, inr);
    ex.get_policy_handler().wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  ex.get_policy_handler().wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  auto gemm_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : gemm_params) {
    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         index_t size) { run<scalar_t>(st, exPtr, size); };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 exPtr, size);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  register_benchmark<float>(args, exPtr);
#ifdef DOUBLE_SUPPORT
  register_benchmark<double>(args, exPtr);
#endif
}
}  // namespace blas_benchmark
