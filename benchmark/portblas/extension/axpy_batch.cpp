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
 *  @filename axpy_batch.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::ExtensionOp benchmark_op =
    blas_benchmark::utils::ExtensionOp::axpy_batch;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         scalar_t alpha, index_t inc_x, index_t inc_y, index_t stride_x_mul,
         index_t stride_y_mul, index_t batch_size, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Google-benchmark counters are double.
  blas_benchmark::utils::init_extension_counters<benchmark_op, scalar_t>(
      state, size, batch_size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  const auto stride_x{size * std::abs(inc_x) * stride_x_mul};
  const auto stride_y{size * std::abs(inc_y) * stride_y_mul};

  const index_t size_x{stride_x * batch_size};
  const index_t size_y{stride_y * batch_size};
  // Create data
  std::vector<scalar_t> vx =
      blas_benchmark::utils::random_data<scalar_t>(size_x);
  std::vector<scalar_t> vy =
      blas_benchmark::utils::random_data<scalar_t>(size_y);

  auto inx = blas::helper::allocate<mem_alloc, scalar_t>(size_x, q);
  auto iny = blas::helper::allocate<mem_alloc, scalar_t>(size_y, q);

  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, vx.data(), inx, size_x);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, vy.data(), iny, size_y);

  sb_handle.wait({copy_x, copy_y});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> y_ref = vy;
  for (auto i = 0; i < batch_size; ++i) {
    reference_blas::axpy(size, static_cast<scalar_t>(alpha),
                         vx.data() + i * stride_x, inc_x,
                         y_ref.data() + i * stride_y, inc_y);
  }
  std::vector<scalar_t> y_temp = vy;
  {
    auto y_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_y, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, y_temp.data(),
                                                            y_temp_gpu, size_y);
    sb_handle.wait(copy_temp);
    auto axpy_batch_event =
        _axpy_batch(sb_handle, size, alpha, inx, inc_x, stride_x, y_temp_gpu,
                    inc_y, stride_y, batch_size);
    sb_handle.wait(axpy_batch_event);
    auto copy_output =
        blas::helper::copy_to_host(q, y_temp_gpu, y_temp.data(), size_y);
    sb_handle.wait(copy_output);

    blas::helper::deallocate<mem_alloc>(y_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(y_temp, y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _axpy_batch(sb_handle, size, alpha, inx, inc_x, stride_x, iny,
                             inc_y, stride_y, batch_size);
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

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(inx, q);
  blas::helper::deallocate<mem_alloc>(iny, q);
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<axpy_batch_param_t<scalar_t>> params) {
  for (auto p : params) {
    index_t n, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size;
    scalar_t alpha;
    std::tie(n, alpha, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size) =
        p;
    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         index_t size, scalar_t alpha, index_t inc_x,
                         index_t inc_y, index_t stride_x_mul,
                         index_t stride_y_mul, index_t batch_size,
                         bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, size, alpha, inc_x, inc_y,
                               stride_x_mul, stride_y_mul, batch_size, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t, index_t>(
            n, alpha, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size,
            mem_type)
            .c_str(),
        BM_lambda, sb_handle_ptr, n, alpha, inc_x, inc_y, stride_x_mul,
        stride_y_mul, batch_size, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto axpy_batch_params =
      blas_benchmark::utils::get_axpy_batch_params<scalar_t>(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER,
      axpy_batch_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM,
      axpy_batch_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
