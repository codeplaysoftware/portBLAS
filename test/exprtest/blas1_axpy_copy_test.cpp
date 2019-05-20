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
#include "operations/blas_operators.hpp"
#include "blas_meta.h"
#include "operations/blas1_trees.hpp"
typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;


TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, axpy_copy_test)
REGISTER_STRD(::RANDOM_STRD, axpy_copy_test)
REGISTER_PREC(float, 1e-4, axpy_copy_test)
REGISTER_PREC(double, 1e-6, axpy_copy_test)
REGISTER_PREC(std::complex<float>, 1e-4, axpy_copy_test)
REGISTER_PREC(std::complex<double>, 1e-6, axpy_copy_test)

TYPED_TEST(BLAS_Test, axpy_copy_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class axpy_copy_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  scalar_t prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // setting alpha to some value
  scalar_t alpha(1.54);
  
  // vector X will be copied to Y
  std::vector<scalar_t> vX(size, 0);
  std::vector<scalar_t> vY(size, 0);

  // vector Z will be copied to W
  std::vector<scalar_t> vZ(size, 0);
  std::vector<scalar_t> vW(size, 0);

  // vectors Y and W will be used in the AXPY operation

  TestClass::set_rand(vX, size);
  TestClass::set_rand(vZ, size);

  // vector A will store the result of AXPY for comparison later
  std::vector<scalar_t> vA(size, 0);


  // First copy X->Y, and Z->W
  for (int i = 0; i < size; ++i) {
    vY[i] = vX[i];
    vW[i] = vZ[i];
  }

  // Now copute the AXPY using the copied vectors Y and W, and store it in vector A
  for (int i = 0; i < size; ++i) {
    if (i % strd == 0) {
      vA[i] = alpha * vY[i] + vW[i];
    } else {
      vA[i] = vW[i];
    }
  }

  auto q = make_queue();
  Executor<ExecutorType> ex(q);

  auto gpu_vX = blas::make_sycl_iterator_buffer<scalar_t>(vX, size);
  auto gpu_vY = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vZ = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vW = ex.get_policy_handler().template allocate<scalar_t>(size);

  ex.get_policy_handler().copy_to_device(vY.data(), gpu_vY, size);
  ex.get_policy_handler().copy_to_device(vW.data(), gpu_vW, size);

  auto buff_vX = ex.get_policy_handler().get_buffer(gpu_vX);
  auto buff_vY = ex.get_policy_handler().get_buffer(gpu_vY);
  auto buff_vZ = ex.get_policy_handler().get_buffer(gpu_vZ);
  auto buff_vW = ex.get_policy_handler().get_buffer(gpu_vW);

  auto view_vX = make_vector_view(ex, buff_vX, strd, (size + strd - 1) / strd);
  auto view_vY = make_vector_view(ex, buff_vY, strd, (size + strd - 1) / strd);
  auto view_vZ = make_vector_view(ex, buff_vZ, strd, (size + strd - 1) / strd);
  auto view_vW = make_vector_view(ex, buff_vW, strd, (size + strd - 1) / strd);

  // First copy operation
  auto copy_x_to_y = make_op<Assign>(view_vX, view_vY);

  // Second copy operation
  auto copy_z_to_w = make_op<Assign>(view_vZ, view_vW);

  // Now axpy the copied vectors
  auto axpy_scal_op = make_op<ScalarOp, ProductOperator>(alpha, copy_x_to_y);
  auto axpy_add_op = make_op<BinaryOp, AddOperator>(copy_z_to_w, axpy_scal_op);
  auto copy_axpy_op_tree = make_op<Assign>(copy_z_to_w, axpy_add_op);

  auto axpy_ev = ex.execute(copy_axpy_op_tree);
  ex.get_policy_handler().wait(axpy_ev);

  // Retrieve the axpy value back to host memory
  auto cpw_event = ex.get_policy_handler().copy_to_host(gpu_vW, vW.data(), size);
  ex.get_policy_handler().wait(cpw_event);

  // check that both results are the same
  for (int i = 0; i < size; ++i) {
    if (i % strd == 0) {
      ASSERT_NEAR(vA[i], vW[i], prec);
    } else {
      ASSERT_EQ(vA[i], vW[i]);
    }
  }
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vY);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vW);


    // Wait for copies 
  /*auto axpy_ev2 = ex.execute(copy_x_to_y);
  ex.get_policy_handler().wait(axpy_ev2);
  auto axpy_ev3 = ex.execute(copy_z_to_w);
  ex.get_policy_handler().wait(axpy_ev3);*/


  /*auto axpy_op_tree = 
    make_op<Assign> 
      (view_vY, make_op<BinaryOp, AddOperator>       // y = 
        (view_vY, make_op<ScalarOp, ProductOperator> // y = y + 
          (alpha, view_vX)                           // y = y + ax
        )
      );

  auto copy_op_tree = 
    make_op<Assign>
      (view_vZ, view_vY); // z = y
*/

/*auto axpy_scal_op = make_op<ScalarOp, ProductOperator>(alpha, view_vY);
  auto axpy_add_op = make_op<BinaryOp, AddOperator>(view_vW, axpy_scal_op);
  auto copy_axpy_op_tree = make_op<Assign>(view_vW, axpy_add_op);*/

}