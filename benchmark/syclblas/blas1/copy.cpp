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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename copy.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size, int incx, int incy) {
  std::ostringstream str{};
  str << "BM_Copy<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << size << "/" << incx << "/" << incy;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         index_t incx, index_t incy, bool* success) {
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 0.0;
  state.counters["bytes_processed"] = size_d * sizeof(scalar_t);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  auto size_x = size * incx;
  auto size_y = size * incy;
  std::vector<scalar_t> x =
      blas_benchmark::utils::random_data<scalar_t>(size_x);
  std::vector<scalar_t> y =
      blas_benchmark::utils::random_data<scalar_t>(size_y);

  auto x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x, size_x);
  auto y_gpu = blas::make_sycl_iterator_buffer<scalar_t>(y, size_y);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> y_ref = y;
  reference_blas::copy(size, x.data(), incx, y_ref.data(), incy);
  std::vector<scalar_t> y_temp = y;
  {
    auto y_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(y_temp, size_y);
    auto event = _copy(sb_handle, size, x_gpu, incx, y_temp_gpu, incy);
    sb_handle.wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(y_temp, y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _copy(sb_handle, size, x_gpu, incx, y_gpu, incy);
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

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto params = blas_benchmark::utils::get_copy_params<scalar_t>(args);

  for (auto p : params) {
    index_t size, incx, incy;
    scalar_t unused;  // Work around a dpcpp compiler bug
    std::tie(size, incx, incy, unused) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         index_t size, index_t incx, index_t incy,
                         bool* success) {
      run<scalar_t>(st, sb_handle_ptr, size, incx, incy, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size, incx, incy).c_str(),
                                 BM_lambda, sb_handle_ptr, size, incx, incy,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
