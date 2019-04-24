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
 *  @filename blas1_rotg_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
/**
 * ROTG.
 * @brief Consturcts given plane rotation
 * Not implemented.
 */
template <typename element_t>
void _rotg(element_t &_alpha, element_t &_beta, element_t &_cos,
           element_t &_sin) {
  element_t abs_alpha = std::abs(_alpha);
  element_t abs_beta = std::abs(_beta);
  element_t roe = (abs_alpha > abs_beta) ? _alpha : _beta;
  element_t scale = abs_alpha + abs_beta;
  element_t norm;
  element_t aux;

  if (scale == constant<element_t, const_val::zero>::value()) {
    _cos = constant<element_t, const_val::one>::value();
    _sin = constant<element_t, const_val::zero>::value();
    norm = constant<element_t, const_val::zero>::value();
    aux = constant<element_t, const_val::zero>::value();
  } else {
    norm = scale * std::sqrt((_alpha / scale) * (_alpha / scale) +
                             (_beta / scale) * (_beta / scale));
    if (roe < constant<element_t, const_val::zero>::value()) norm = -norm;
    _cos = _alpha / norm;
    _sin = _beta / norm;
    if (abs_alpha > abs_beta) {
      aux = _sin;
    } else if (_cos != constant<element_t, const_val::zero>::value()) {
      aux = constant<element_t, const_val::one>::value() / _cos;
    } else {
      aux = constant<element_t, const_val::one>::value();
    }
  }
  _alpha = norm;
  _beta = aux;
}

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(2, rotg_test)
REGISTER_STRD(1, rotg_test)
REGISTER_PREC(float, 1e-4, rotg_test)
REGISTER_PREC(double, 1e-7, rotg_test)

TYPED_TEST(BLAS_Test, rotg_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class rotg_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  scalar_t prec = TestClass::template test_prec<test>();

  std::vector<scalar_t> vX(size);
  std::vector<scalar_t> vY(size);
  std::vector<scalar_t> vR(1, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  SYCL_DEVICE_SELECTOR d;
  scalar_t _cos, _sin;

  scalar_t giv = 0;
  // givens rotation of vectors vX and vY
  // and computation of dot of both vectors
  for (int i = 0; i < size; i += strd) {
    scalar_t x = vX[i], y = vY[i];
    if (i == 0) {
      // compute _cos and _sin
      _rotg(x, y, _cos, _sin);
    }
    x = vX[i], y = vY[i];
    giv += ((x * _cos + y * _sin) * (y * _cos - x * _sin));
  }

  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);

  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vY = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vR = ex.get_policy_handler().template allocate<scalar_t>(1);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vY.data(), gpu_vY, size);
  ex.get_policy_handler().copy_to_device(vR.data(), gpu_vR, 1);
  _rot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, _cos, _sin);
  _dot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);

  // check that the result is the same
  ASSERT_NEAR(giv, vR[0], prec);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vX);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vY);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vR);
}
