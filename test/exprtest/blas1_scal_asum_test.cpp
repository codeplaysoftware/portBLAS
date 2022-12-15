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
 *  @filename blas1_scal_asum_test.cpp
 *
 *
 **************************************************************************/
#include "blas_test.hpp"
#include "sycl_blas.hpp"

// inputs combination
template <typename scalar_t>
using combination_t = std::tuple<int, scalar_t, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  scalar_t alpha;
  int incX;
  std::tie(size, alpha, incX) = combi;

  // Dimensions of input vector x
  int x_dim = size * incX;

  // SCAL Input vectors x and cpu_x,
  std::vector<scalar_t> v_x(x_dim);
  fill_random(v_x);
  std::vector<scalar_t> v_cpu_x = v_x;

  // ASUM output y and cpu_y
  std::vector<scalar_t> v_y(1, scalar_t(0));
  scalar_t cpu_y = 0;

  // Reference BLAS implementation
  reference_blas::scal(size, alpha, v_cpu_x.data(), incX);
  cpu_y = reference_blas::asum(size, v_cpu_x.data(), incX);

  // SYCL-BLAS implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_dim);
  auto gpu_y_v = blas::make_sycl_iterator_buffer<scalar_t>(1);

  // Copy input data from host to device
  auto xcp_ev = blas::helper::copy_to_device(sb_handle.get_queue(), v_x.data(),
                                             gpu_x_v, x_dim);
  sb_handle.wait(xcp_ev);

  // Dimensions of vector view for ASUM operations
  int view_x_dim = (x_dim + incX - 1) / incX;

  // Views
  auto view_x = make_vector_view(gpu_x_v, incX, view_x_dim);
  auto view_assign_x = make_vector_view(gpu_x_v, 1, x_dim);
  auto view_y = make_vector_view(gpu_y_v, 1, 1);

  // Assign reduction parameters
  const auto localSize = sb_handle.get_work_group_size();
  const auto nWG = 2 * localSize;

  // SCAL expressions
  auto scal_op = make_op<ScalarOp, ProductOperator>(alpha, view_x);
  auto scal_assign_op = make_op<Assign>(view_x, scal_op);
  auto asum_op = make_AssignReduction<AbsoluteAddOperator>(
      view_y, scal_assign_op, localSize, localSize * nWG);

  // Execute the SCAL+ASUM tree
  auto event = sb_handle.execute(asum_op);
  sb_handle.wait(event);

  // Copy the result back to host memory
  auto getResultEv =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_y_v, v_y.data(), 1);
  sb_handle.wait(getResultEv);

  ASSERT_TRUE(utils::almost_equal(cpu_y, v_y[0]));
}

template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(16, 1023),   // size
                                      ::testing::Values(0.0, 1.34),  // alpha
                                      ::testing::Values(1, 4));      // incX

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX;
  T alpha;
  BLAS_GENERATE_NAME(info.param, size, alpha, incX);
}

BLAS_REGISTER_TEST_FLOAT(ScalAsumTree, combination_t, combi, generate_name);
