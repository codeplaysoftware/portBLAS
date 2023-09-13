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
 *  @filename asum.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Asum<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::asum, scalar_t>(state, size);

  using data_t = scalar_t;

  // Create data
  std::vector<data_t> v1 = blas_benchmark::utils::random_data<data_t>(size);
  scalar_t vr;

  // Device vectors
  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(executorPtr, v1.data(), size);
  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufr(executorPtr, &vr, 1);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  data_t vr_ref = reference_blas::asum(size, v1.data(), 1);
  data_t vr_temp = 0;
  {
    MemBuffer<scalar_t, CL_MEM_READ_ONLY> vr_temp_gpu(executorPtr, &vr_temp, 1);
    cl_event event;
    clblast::Asum<scalar_t>(size, vr_temp_gpu.dev(), 0, buf1.dev(), 0, 1,
                            executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
  }

  if (!utils::almost_equal(vr_temp, vr_ref)) {
    std::ostringstream err_stream;
    err_stream << "Value mismatch: " << vr_temp << "; expected " << vr_ref;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Asum<scalar_t>(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                            executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
    return {event};
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
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto asum_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : asum_params) {
    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         index_t size, bool* success) {
      run<scalar_t>(st, exPtr, size, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 exPtr, size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
