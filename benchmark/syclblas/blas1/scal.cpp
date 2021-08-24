/**************************************************************************
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
 *  @filename scal.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Scal<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);

  ExecutorType& ex = *executorPtr;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Create data
  std::vector<data_t> v1 = blas_benchmark::utils::random_data<data_t>(size);
  auto alpha = blas_benchmark::utils::random_scalar<scalar_t>();

#ifdef SYCL_BLAS_USE_USM
  data_t* in = cl::sycl::malloc_device<data_t>(size, ex.get_policy_handler().get_queue());

  ex.get_policy_handler().get_queue().memcpy(in, v1.data(), sizeof(data_t) * size).wait();
#else
  auto in = utils::make_quantized_buffer<scalar_t>(ex, v1);
#endif

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<data_t> v1_ref = v1;
  reference_blas::scal(size, static_cast<data_t>(alpha), v1_ref.data(), 1);
  std::vector<data_t> v1_temp = v1;
  {
    auto v1_temp_gpu = 
#ifdef SYCL_BLAS_USE_USM
        cl::sycl::malloc_device<data_t>(size, ex.get_policy_handler().get_queue());
    ex.get_policy_handler().get_queue().memcpy(v1_temp_gpu, v1_temp.data(), sizeof(data_t) * size).wait();
#else
        utils::make_quantized_buffer<scalar_t>(ex, v1_temp);
#endif

    auto ev = _scal(ex, size, alpha, v1_temp_gpu, 1);
#ifdef SYCL_BLAS_USE_USM
    ex.get_policy_handler().wait(ev);
#endif

    auto event =
#ifdef SYCL_BLAS_USE_USM
        ex.get_policy_handler().get_queue().memcpy(v1_temp.data(), v1_temp_gpu, sizeof(data_t) * size);
#else
        utils::quantized_copy_to_host<scalar_t>(ex, v1_temp_gpu, v1_temp);
#endif
    ex.get_policy_handler().wait({event});

#ifdef SYCL_BLAS_USE_USM
    cl::sycl::free(v1_temp_gpu, ex.get_policy_handler().get_queue());
#endif
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v1_temp, v1_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _scal(ex, size, alpha, in, 1);
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
  };

  blas_benchmark::utils::calc_avg_counters(state);

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(in, ex.get_policy_handler().get_queue());
#endif
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : gemm_params) {
    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         index_t size, bool* success) {
      run<scalar_t>(st, exPtr, size, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 exPtr, size, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
