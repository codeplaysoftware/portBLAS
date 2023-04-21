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
 *  @filename reduction_test.cpp
 *
 **************************************************************************/

#include <limits>

#include "blas_test.hpp"

enum operator_t : int {
  Add = 0,
  Product = 1,
  Max = 2,
  Min = 3,
  AbsoluteAdd = 4,
  Mean = 5,
};

using index_t = int;

template <typename scalar_t>
using combination_t =
    std::tuple<index_t, index_t, index_t, operator_t, reduction_dim_t>;

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values(1, 7, 513),                // rows
    ::testing::Values(1, 15, 1000, 1337, 8195),  // columns
    ::testing::Values(1, 2, 3),                  // ld_mul
    ::testing::Values(operator_t::Add, operator_t::Max, operator_t::Min,
                      operator_t::AbsoluteAdd, operator_t::Mean,
                      operator_t::Product),
    ::testing::Values(reduction_dim_t::inner, reduction_dim_t::outer));

template <>
inline void dump_arg<operator_t>(std::ostream& ss, operator_t op) {
  ss << (int)op;
}

template <>
inline void dump_arg<reduction_dim_t>(std::ostream& ss,
                                      reduction_dim_t reductionDim) {
  ss << (int)reductionDim;
}

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  index_t rows, cols, ldMul;
  operator_t op;
  reduction_dim_t reductionDim;
  BLAS_GENERATE_NAME(info.param, rows, cols, ldMul, op, reductionDim);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t rows, cols, ld_mul;
  operator_t op;
  reduction_dim_t reduction_dim;
  std::tie(rows, cols, ld_mul, op, reduction_dim) = combi;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  index_t ld = rows * ld_mul;

  std::vector<scalar_t> in_m(ld * cols);
  const auto out_size = reduction_dim == reduction_dim_t::outer ? rows : cols;
  std::vector<scalar_t> out_v_gpu(out_size);
  std::vector<scalar_t> out_v_cpu(out_size);

  if (op == operator_t::Product) {
    // Use smaller input range for Product tests since the product
    // operation saturates float overflow faster than the other operations
    fill_random_with_range(in_m, scalar_t{-2}, scalar_t{1});
  } else {
    fill_random(in_m);
  }

  scalar_t init_val;
  switch (op) {
    case operator_t::Add:
    case operator_t::AbsoluteAdd:
    case operator_t::Mean:
      init_val = scalar_t{0};
      break;
    case operator_t::Product:
      init_val = scalar_t{1};
      break;
    case operator_t::Min:
      init_val = std::numeric_limits<scalar_t>::max();
      break;
    case operator_t::Max:
      init_val = std::numeric_limits<scalar_t>::lowest();
      break;
  }

  /* Reduction function. */
  std::function<scalar_t(scalar_t, scalar_t)> reduction_func;
  switch (op) {
    case operator_t::Add:
    case operator_t::Mean:
      reduction_func = [=](scalar_t l, scalar_t r) -> scalar_t {
        return l + r;
      };
      break;
    case operator_t::AbsoluteAdd:
      reduction_func = [=](scalar_t l, scalar_t r) -> scalar_t {
        return std::abs(l) + std::abs(r);
      };
      break;
    case operator_t::Product:
      reduction_func = [=](scalar_t l, scalar_t r) -> scalar_t {
        return l * r;
      };
      break;
    case operator_t::Min:
      reduction_func = [=](scalar_t l, scalar_t r) -> scalar_t {
        return l < r ? l : r;
      };
      break;
    case operator_t::Max:
      reduction_func = [=](scalar_t l, scalar_t r) -> scalar_t {
        return l > r ? l : r;
      };
      break;
  }

  /* Reduce the reference by hand */
  if (reduction_dim == reduction_dim_t::outer) {
    for (index_t i = 0; i < rows; i++) {
      out_v_cpu[i] = init_val;
      out_v_gpu[i] = init_val;
      for (index_t j = 0; j < cols; j++) {
        out_v_cpu[i] = reduction_func(out_v_cpu[i], in_m[ld * j + i]);
      }
    }
  } else if (reduction_dim == reduction_dim_t::inner) {
    for (index_t i = 0; i < cols; i++) {
      out_v_cpu[i] = init_val;
      out_v_gpu[i] = init_val;
      for (index_t j = 0; j < rows; j++) {
        out_v_cpu[i] = reduction_func(out_v_cpu[i], in_m[ld * i + j]);
      }
    }
  }

  if (op == operator_t::Mean) {
    const auto nelems = reduction_dim == reduction_dim_t::outer ? cols : rows;
    std::transform(out_v_cpu.begin(), out_v_cpu.end(), out_v_cpu.begin(),
                   [=](scalar_t val) -> scalar_t {
                     return val / static_cast<scalar_t>(nelems);
                   });
  }

  auto m_in_gpu = blas::make_sycl_iterator_buffer<scalar_t>(in_m, ld * cols);
  auto v_out_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(out_v_gpu, out_size);

  blas::SB_Handle::event_t ev;
  try {
    switch (op) {
      case operator_t::Add:
        ev = extension::_reduction<AddOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
      case operator_t::Product:
        ev = extension::_reduction<ProductOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
      case operator_t::Max:
        ev = extension::_reduction<MaxOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
      case operator_t::Min:
        ev = extension::_reduction<MinOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
      case operator_t::AbsoluteAdd:
        ev = extension::_reduction<AbsoluteAddOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
      case operator_t::Mean:
        ev = extension::_reduction<MeanOperator, scalar_t>(
            sb_handle, m_in_gpu, ld, v_out_gpu, rows, cols, reduction_dim);
        break;
    }
  } catch (cl::sycl::exception& e) {
    std::cerr << "Exception occured:" << std::endl;
    std::cerr << e.what() << std::endl;
  }
  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), v_out_gpu, out_v_gpu.data(), out_size);
  sb_handle.wait(event);

  ASSERT_TRUE(utils::compare_vectors(out_v_gpu, out_v_cpu));
}

BLAS_REGISTER_TEST_ALL(ReductionPartial, combination_t, combi, generate_name);
