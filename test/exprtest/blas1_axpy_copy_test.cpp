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
 *  @filename blas1_axpy_copy_test.cpp
 *
 **************************************************************************/
#include "blas_test.hpp"
#include "sycl_blas.hpp"

// inputs combination
template <typename scalar_t>
using combination_t = std::tuple<int, scalar_t, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  scalar_t alpha;
  int incX;
  int incY;
  std::tie(size, alpha, incX, incY) = combi;

  // Dimensions of input vectors x and y
  int x_dim = size * incX;
  int y_dim = size * incY;

  // Input vectors x and y (y is also output of AXPY)
  std::vector<scalar_t> v_x(x_dim);
  std::vector<scalar_t> v_y(y_dim);
  fill_random(v_x);
  fill_random(v_y);

  // Copy of vector x
  std::vector<scalar_t> v_xcopy(x_dim);

  // Copy of vector y
  // Used to store output on host for later comparison with SYCL result
  std::vector<scalar_t> v_cpu_y(y_dim);

  // Reference BLAS implementation
  reference_blas::copy(size, v_x.data(), incX, v_xcopy.data(), incX);
  reference_blas::copy(y_dim, v_y.data(), 1, v_cpu_y.data(), 1);
  reference_blas::axpy(size, alpha, v_xcopy.data(), incX, v_cpu_y.data(), incY);

  // SYCL-BLAS implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_dim);
  auto gpu_y_v = blas::make_sycl_iterator_buffer<scalar_t>(y_dim);
  auto gpu_xcopy_v = blas::make_sycl_iterator_buffer<scalar_t>(x_dim);
  auto gpu_ycopy_v = blas::make_sycl_iterator_buffer<scalar_t>(y_dim);

  // Copy input data from host to device
  auto xcp_ev = blas::helper::copy_to_device(sb_handle.get_queue(), v_x.data(),
                                             gpu_x_v, x_dim);
  sb_handle.wait(xcp_ev);
  auto ycp_ev = blas::helper::copy_to_device(sb_handle.get_queue(), v_y.data(),
                                             gpu_y_v, y_dim);
  sb_handle.wait(ycp_ev);

  // Dimensions of vector view for AXPY operations
  int view_x_dim = (x_dim + incX - 1) / incX;
  int view_y_dim = (y_dim + incY - 1) / incY;

  // Views
  // - for axpy operation where arbitrary stride is used
  auto view_x_incX = make_vector_view(gpu_x_v, incX, view_x_dim);
  auto view_y_incY = make_vector_view(gpu_y_v, incY, view_y_dim);

  // - for copy operations where we want stride = 1
  auto view_xcopy_inc1 = make_vector_view(gpu_xcopy_v, 1, x_dim);
  auto view_ycopy_inc1 = make_vector_view(gpu_ycopy_v, 1, y_dim);

  // Expressions to copy from device to device
  auto xCopyOp = make_op<Assign>(view_xcopy_inc1, view_x_incX);
  auto yCopyOp = make_op<Assign>(view_ycopy_inc1, view_y_incY);

  // AXPY expressions
  auto axpy_scal_op = make_op<ScalarOp, ProductOperator>(alpha, xCopyOp);
  auto axpy_add_op = make_op<BinaryOp, AddOperator>(yCopyOp, axpy_scal_op);
  auto copy_axpy_op_tree = make_op<Assign>(view_y_incY, axpy_add_op);

  // Execute the COPY+AXPY tree
  auto axpy_event = sb_handle.execute(copy_axpy_op_tree);
  sb_handle.wait(axpy_event);

  // Copy the result back to host memory
  auto getResultEv = blas::helper::copy_to_host(sb_handle.get_queue(), gpu_y_v,
                                                v_y.data(), y_dim);
  sb_handle.wait(getResultEv);

  ASSERT_TRUE(utils::compare_vectors(v_cpu_y, v_y));
}

template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values(16, 1023),             // size
                       ::testing::Values<scalar_t>(0.0, 1.34),  // alpha
                       ::testing::Values(1, 4),                 // incX
                       ::testing::Values(1, 3));                // incY

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX, incY;
  T alpha;
  BLAS_GENERATE_NAME(info.param, size, alpha, incX, incY);
}

BLAS_REGISTER_TEST_FLOAT(AxpyCopyTree, combination_t, combi, generate_name);
