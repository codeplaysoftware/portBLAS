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
 *  @filename copy.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::copy;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         index_t incx, index_t incy, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::copy, scalar_t>(state, size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  auto size_x = size * incx;
  auto size_y = size * incy;
  std::vector<scalar_t> x =
      blas_benchmark::utils::random_data<scalar_t>(size_x);
  std::vector<scalar_t> y =
      blas_benchmark::utils::random_data<scalar_t>(size_y);

  auto x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_x, q);
  auto y_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_y, q);

  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, x.data(), x_gpu, size_x);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, y.data(), y_gpu, size_y);

  sb_handle.wait({copy_x, copy_y});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> y_ref = y;
  reference_blas::copy(size, x.data(), incx, y_ref.data(), incy);
  std::vector<scalar_t> y_temp = y;
  {
    auto y_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_y, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, y_temp.data(),
                                                            y_temp_gpu, size_y);
    sb_handle.wait(copy_temp);
    auto copy_event = blas::_copy<blas::SB_Handle, index_t, decltype(x_gpu),
                                  decltype(y_temp_gpu), index_t>(
        sb_handle, size, x_gpu, incx, y_temp_gpu, incy);
    sb_handle.wait(copy_event);

    auto copy_out = blas::helper::copy_to_host<scalar_t>(q, y_temp_gpu,
                                                         y_temp.data(), size_y);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(y_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(y_temp, y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<sycl::event> {
    auto event =
        blas::_copy<blas::SB_Handle, index_t, decltype(x_gpu), decltype(y_gpu),
                    index_t>(sb_handle, size, x_gpu, incx, y_gpu, incy);
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

  state.SetItemsProcessed(0);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(x_gpu, q);
  blas::helper::deallocate<mem_alloc>(y_gpu, q);
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<copy_param_t<scalar_t>> params) {
  for (auto p : params) {
    index_t size, incx, incy;
    scalar_t unused;  // Work around a dpcpp compiler bug
                      // (https://github.com/intel/llvm/issues/7075)
    std::tie(size, incx, incy, unused) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         index_t size, index_t incx, index_t incy,
                         bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, size, incx, incy, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            size, incx, incy, mem_type).c_str(),
        BM_lambda, sb_handle_ptr, size, incx, incy, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto copy_params = blas_benchmark::utils::get_copy_params<scalar_t>(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER, copy_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, copy_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
