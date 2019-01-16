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
template <typename T>
void _rotg(T &_alpha, T &_beta, T &_cos, T &_sin) {
  T abs_alpha = std::abs(_alpha);
  T abs_beta = std::abs(_beta);
  T roe = (abs_alpha > abs_beta) ? _alpha : _beta;
  T scale = abs_alpha + abs_beta;
  T norm;
  T aux;

  if (scale == constant<T, const_val::zero>::value) {
    _cos = constant<T, const_val::one>::value;
    _sin = constant<T, const_val::zero>::value;
    norm = constant<T, const_val::zero>::value;
    aux = constant<T, const_val::zero>::value;
  } else {
    norm = scale * std::sqrt((_alpha / scale) * (_alpha / scale) +
                             (_beta / scale) * (_beta / scale));
    if (roe < constant<T, const_val::zero>::value) norm = -norm;
    _cos = _alpha / norm;
    _sin = _beta / norm;
    if (abs_alpha > abs_beta) {
      aux = _sin;
    } else if (_cos != constant<T, const_val::zero>::value) {
      aux = constant<T, const_val::one>::value / _cos;
    } else {
      aux = constant<T, const_val::one>::value;
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
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class rotg_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vR(1, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  SYCL_DEVICE_SELECTOR d;
  ScalarT _cos, _sin;

  ScalarT giv = 0;
  // givens rotation of vectors vX and vY
  // and computation of dot of both vectors
  for (int i = 0; i < size; i += strd) {
    ScalarT x = vX[i], y = vY[i];
    if (i == 0) {
      // compute _cos and _sin
      _rotg(x, y, _cos, _sin);
    }
    x = vX[i], y = vY[i];
    giv += ((x * _cos + y * _sin) * (y * _cos - x * _sin));
  }

  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);

  auto gpu_vX = ex.get_policy_handler().template allocate<ScalarT>(size);
  auto gpu_vY = ex.get_policy_handler().template allocate<ScalarT>(size);
  auto gpu_vR = ex.get_policy_handler().template allocate<ScalarT>(1);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vY.data(), gpu_vY, size);
  ex.get_policy_handler().copy_to_device(vR.data(), gpu_vR, 1);
  _rot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, _cos, _sin);
  _dot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);

  // check that the result is the same
  ASSERT_NEAR(giv, vR[0], prec);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vX);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vY);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vR);
}
