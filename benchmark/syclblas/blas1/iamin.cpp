/**************************************************************************
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
 *  @filename iamin.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size, std::string mem_type) {
  std::ostringstream str{};
  str << "BM_Iamin<" << blas_benchmark::utils::get_type_name<scalar_t>();
  str << ">/" << mem_type << "/" << size;
  return str.str();
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::iamin, scalar_t>(state, size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  using tuple_scalar_t = blas::IndexValueTuple<index_t, scalar_t>;

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  tuple_scalar_t out{0, 0};

  std::transform(std::begin(v1), std::end(v1), std::begin(v1), [](scalar_t v) {
    return utils::clamp_to_limits<scalar_t>(v);
  });

  typename blas::helper::AllocHelper<scalar_t, mem_alloc>::type inx;
  typename blas::helper::AllocHelper<tuple_scalar_t, mem_alloc>::type outI;
  cl::sycl::event copy_x, copy_outI;

  std::tie(inx, copy_x) =
      blas::helper::allocate<mem_alloc, scalar_t>(v1.data(), size, q);
  std::tie(outI, copy_outI) =
      blas::helper::allocate<mem_alloc, tuple_scalar_t>(&out, 1, q);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  index_t idx_ref =
      static_cast<index_t>(reference_blas::iamin(size, v1.data(), 1));
  tuple_scalar_t idx_temp{-1, -1};
  {
    typename blas::helper::AllocHelper<tuple_scalar_t, mem_alloc>::type
        idx_temp_gpu;
    cl::sycl::event copy_temp;
    std::tie(idx_temp_gpu, copy_temp) =
        blas::helper::allocate<mem_alloc, tuple_scalar_t>(&idx_temp, 1, q);
    auto event = _iamin(sb_handle, size, inx, static_cast<index_t>(1),
                        idx_temp_gpu, {copy_x, copy_temp});
    sb_handle.wait(event);
    auto copy_output =
        blas::helper::copy_to_host(q, idx_temp_gpu, &idx_temp, 1);
    sb_handle.wait(copy_output);

    blas::helper::deallocate<mem_alloc>(idx_temp_gpu, q);
  }

  if (idx_temp.ind != idx_ref) {
    std::ostringstream err_stream;
    err_stream << "Index mismatch: " << idx_temp.ind << "; expected "
               << idx_ref;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _iamin(sb_handle, size, inx, static_cast<index_t>(1), outI,
                        {copy_x, copy_outI});
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
  blas::helper::deallocate<mem_alloc>(outI, q);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto iamin_params = blas_benchmark::utils::get_blas1_params(args);

#define REGISTER_MEM_TYPE_BENCHMARKS(MEMORY, NAME)                             \
  for (auto size : iamin_params) {                                             \
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
