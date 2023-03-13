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
 *  @filename axpy.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size, std::string mem_type) {
  std::ostringstream str{};
  str << "BM_Axpy<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << mem_type << "/" << size;
  return str.str();
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::axpy, scalar_t>(state, size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = blas_benchmark::utils::random_data<scalar_t>(size);
  auto alpha = blas_benchmark::utils::random_scalar<scalar_t>();

  typename blas::helper::AllocHelper<scalar_t, mem_alloc>::type inx, iny;
  cl::sycl::event copy_x, copy_y;

  std::tie(inx, copy_x) =
      blas::helper::allocate<mem_alloc, scalar_t>(v1.data(), size, q);
  std::tie(iny, copy_y) =
      blas::helper::allocate<mem_alloc, scalar_t>(v2.data(), size, q);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> y_ref = v2;
  reference_blas::axpy(size, static_cast<scalar_t>(alpha), v1.data(), 1,
                       y_ref.data(), 1);
  std::vector<scalar_t> y_temp = v2;
  {
    typename blas::helper::AllocHelper<scalar_t, mem_alloc>::type y_temp_gpu;
    cl::sycl::event copy_temp_y;
    std::tie(y_temp_gpu, copy_temp_y) =
        blas::helper::allocate<mem_alloc, scalar_t>(y_temp.data(), size, q);
    auto event = _axpy(sb_handle, size, alpha, inx, 1, y_temp_gpu, 1,
                       {copy_x, copy_temp_y});
    sb_handle.wait(event);
    auto copy_output =
        blas::helper::copy_to_host(q, y_temp_gpu, y_temp.data(), size);
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
    auto event = _axpy(sb_handle, size, alpha, inx, static_cast<index_t>(1),
                       iny, static_cast<index_t>(1), {copy_x, copy_y});
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

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto axpy_params = blas_benchmark::utils::get_blas1_params(args);

#define REGISTER_MEM_TYPE_BENCHMARKS(MEMORY, NAME)                             \
  for (auto size : axpy_params) {                                              \
    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr, \
                         index_t size, bool* success) {                        \
      run<scalar_t, MEMORY>(st, sb_handle_ptr, size, success);                 \
    };                                                                         \
    benchmark::RegisterBenchmark(get_name<scalar_t>(size, NAME).c_str(),       \
                                 BM_lambda, sb_handle_ptr, size, success)      \
        ->UseRealTime();                                                       \
  }

  REGISTER_MEM_TYPE_BENCHMARKS(blas::helper::AllocType::usm, "usm");
  REGISTER_MEM_TYPE_BENCHMARKS(blas::helper::AllocType::buffer, "buffer");

#undef REGISTER_MEM_TYPE_BENCHMARKS
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
