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
 *  @filename extension_reduction_partial_rows_test.cpp
 *
 **************************************************************************/

#include <limits>

#include "blas_test.hpp"
#include "sycl_blas.hpp"

enum operator_t : int {
  Add = 0,
  Product = 1,
  Division = 2,
  Max = 3,
  Min = 4,
  AbsoluteAdd = 5
};

using index_t = int;

template <typename scalar_t>
using combination_t = std::tuple<index_t, index_t, index_t, operator_t>;

/* Note: the product and division are not tested because our random data may
 * contain values close to zero */
const auto combi = ::testing::Combine(
    ::testing::Values(1, 7, 513),                // rows
    ::testing::Values(1, 15, 1000, 1337, 8195),  // columns
    ::testing::Values(1, 2, 3),                  // ld_mul
    ::testing::Values(operator_t::Add, operator_t::Max, operator_t::Min,
                      operator_t::AbsoluteAdd));

template <typename operator_t, typename scalar_t, typename executor_t,
          typename input_t, typename output_t>
void launch_reduction(executor_t& ex, input_t buffer_in, output_t buffer_out,
                      index_t rows, index_t cols) {
  blas::Reduction<operator_t, input_t, output_t, 64, 256, scalar_t,
                  static_cast<int>(Reduction_t::partial_rows)>
      reduction(buffer_in, buffer_out, rows, cols);
  auto ev = ex.execute(reduction);
#ifdef SYCL_BLAS_USE_USM
  ex.get_policy_handler().wait(ev);
#endif
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t rows, cols, ld_mul;
  operator_t op;
  std::tie(rows, cols, ld_mul, op) = combi;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  index_t ld = rows * ld_mul;

  std::vector<data_t> in_m(ld * cols);
  std::vector<data_t> out_v_gpu(rows);
  std::vector<data_t> out_v_cpu(rows);

  fill_random(in_m);
  for (index_t i = 0; i < rows; i++) {
    out_v_gpu[i] = -1;
  }
  std::copy(out_v_gpu.begin(), out_v_gpu.end(), out_v_cpu.begin());

  /* Initialization value of the reduction accumulators. */
  scalar_t init_val;
  switch (op) {
    case operator_t::Add:
    case operator_t::AbsoluteAdd:
      init_val = 0;
      break;
    case operator_t::Product:
    case operator_t::Division:
      init_val = 1;
      break;
    case operator_t::Min:
      init_val = std::numeric_limits<scalar_t>::max();
      break;
    case operator_t::Max:
      init_val = std::numeric_limits<scalar_t>::min();
      break;
  }

  /* Reduction function. */
  std::function<data_t(data_t, data_t)> reduction_func;
  switch (op) {
    case operator_t::Add:
      reduction_func = [=](data_t l, data_t r) -> data_t { return l + r; };
      break;
    case operator_t::AbsoluteAdd:
      reduction_func = [=](data_t l, data_t r) -> data_t {
        return abs(l) + abs(r);
      };
      break;
    case operator_t::Product:
      reduction_func = [=](data_t l, data_t r) -> data_t {
        return abs(l) * abs(r);
      };
      break;
    case operator_t::Division:
      reduction_func = [=](data_t l, data_t r) -> data_t {
        return abs(l) / abs(r);
      };
      break;
    case operator_t::Min:
      reduction_func = [=](data_t l, data_t r) -> data_t {
        return l < r ? l : r;
      };
      break;
    case operator_t::Max:
      reduction_func = [=](data_t l, data_t r) -> data_t {
        return l > r ? l : r;
      };
      break;
  }

  /* Reduce the reference by hand */
  for (index_t i = 0; i < rows; i++) {
    out_v_cpu[i] = init_val;
    for (index_t j = 0; j < cols; j++) {
      out_v_cpu[i] = reduction_func(out_v_cpu[i], in_m[ld * j + i]);
    }
  }

#ifdef SYCL_BLAS_USE_USM
  data_t* m_in_gpu = cl::sycl::malloc_device<data_t>(ld * cols, q);
  data_t* v_out_gpu = cl::sycl::malloc_device<data_t>(rows, q);

  q.memcpy(m_in_gpu, in_m.data(), sizeof(data_t) * ld * cols).wait();
  q.memcpy(v_out_gpu, out_v_gpu.data(), sizeof(data_t) * rows).wait();
#else
  auto m_in_gpu = utils::make_quantized_buffer<scalar_t>(ex, in_m);
  auto v_out_gpu = utils::make_quantized_buffer<scalar_t>(ex, out_v_gpu);
#endif
  auto buffer_in = make_matrix_view<col_major>(ex, m_in_gpu, rows, cols, ld);
  auto buffer_out = make_matrix_view<col_major>(ex, v_out_gpu, rows, 1, rows);
  try {
    switch (op) {
      case operator_t::Add:
        launch_reduction<AddOperator, scalar_t>(ex, buffer_in, buffer_out, rows,
                                                cols);
        break;
      case operator_t::Product:
        launch_reduction<ProductOperator, scalar_t>(ex, buffer_in, buffer_out,
                                                    rows, cols);
        break;
      case operator_t::Division:
        launch_reduction<DivisionOperator, scalar_t>(ex, buffer_in, buffer_out,
                                                     rows, cols);
        break;
      case operator_t::Max:
        launch_reduction<MaxOperator, scalar_t>(ex, buffer_in, buffer_out, rows,
                                                cols);
        break;
      case operator_t::Min:
        launch_reduction<MinOperator, scalar_t>(ex, buffer_in, buffer_out, rows,
                                                cols);
        break;
      case operator_t::AbsoluteAdd:
        launch_reduction<AbsoluteAddOperator, scalar_t>(ex, buffer_in,
                                                        buffer_out, rows, cols);
        break;
    }
  } catch (cl::sycl::exception& e) {
    std::cerr << "Exception occured:" << std::endl;
    std::cerr << e.what() << std::endl;
  }
  auto event =
#ifdef SYCL_BLAS_USE_USM
      q.memcpy(out_v_gpu.data(), v_out_gpu, sizeof(data_t) * rows);
#else
      utils::quantized_copy_to_host<scalar_t>(ex, v_out_gpu, out_v_gpu);
#endif
  ex.get_policy_handler().wait({event});

  ASSERT_TRUE(utils::compare_vectors(out_v_gpu, out_v_cpu));

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(m_in_gpu, q);
  cl::sycl::free(v_out_gpu, q);
#endif
}

BLAS_REGISTER_TEST(ReductionPartialRows, combination_t, combi);
