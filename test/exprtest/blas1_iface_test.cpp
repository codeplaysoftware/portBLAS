/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas1_iface_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_args<double>> BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, interface1_test)
REGISTER_STRD(1, interface1_test)
REGISTER_PREC(float, 1e-4, interface1_test)
REGISTER_PREC(double, 1e-6, interface1_test)

TYPED_TEST(BLAS_Test, interface1_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class interface1_test;

  size_t size = 25;  // TestClass::template test_size<test>();
  size_t strd = 1;   // TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // creating three random vectors
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vZ(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);
  TestClass::set_rand(vZ, size);

  // the values will first be computed in a for loop:
  size_t imax = 0, imin = 0;
  ScalarT asum(0);
  const ScalarT alpha(0.432);
  ScalarT dot(0);
  ScalarT nrmX(0);
  ScalarT nrmY(0);
  ScalarT max(0);
  ScalarT min(std::numeric_limits<ScalarT>::max());
  ScalarT diff(0);
  ScalarT _cos(0);
  ScalarT _sin(0);
  ScalarT giv(0);
  for (size_t i = 0; i < size; i += strd) {
    ScalarT &x = vX[i];
    ScalarT &y = vY[i];
    ScalarT &z = vZ[i];

    // axpy:
    z = x * alpha + y;

    // reductions;
    asum += std::abs(z);
    dot += x * z;
    nrmX += x * x, nrmY += z * z;
    // iamax
    if (std::abs(z) > std::abs(max)) {
      max = z;
      imax = i;
    }
    // iamin
    if (std::abs(z) < std::abs(min)) {
      min = z;
      imin = i;
    }
    // givens rotation
    if (i == 0) {
      ScalarT n1 = x, n2 = z;
      _rotg(n1, n2, _cos, _sin);
      diff = (z * _cos - x * _sin) - (x * _cos + z * _sin);
    } else if (i == size - 1) {
      diff += (z * _cos - x * _sin) - (x * _cos + z * _sin);
    }
    giv += ((x * _cos + z * _sin) * (z * _cos - x * _sin));
  }
  nrmX = std::sqrt(nrmX), nrmY = std::sqrt(nrmY);

  // creating vectors which will contain the result
  // for asum:
  std::vector<ScalarT> vR(1);
  // for dot:
  std::vector<ScalarT> vS(1);
  // for nrm2:
  std::vector<ScalarT> vT(1);
  // for dot after _rot
  std::vector<ScalarT> vU(1);
  // for iamax/iamin
  std::vector<IndVal<ScalarT>> vImax(
      1, constant<IndVal<ScalarT>, const_val::imax>::value);
  std::vector<IndVal<ScalarT>> vImin(
      1, constant<IndVal<ScalarT>, const_val::imin>::value);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.template allocate<ScalarT>(size);
  auto gpu_vY = ex.template allocate<ScalarT>(size);
  auto gpu_vR = ex.template allocate<ScalarT>(1);
  auto gpu_vS = ex.template allocate<ScalarT>(1);
  auto gpu_vT = ex.template allocate<ScalarT>(1);
  auto gpu_vU = ex.template allocate<ScalarT>(1);
  auto gpu_vImax = ex.template allocate<IndVal<ScalarT>>(1);
  auto gpu_vImin = ex.template allocate<IndVal<ScalarT>>(1);
  ex.copy_to_device(vX.data(), gpu_vX, size);
  ex.copy_to_device(vY.data(), gpu_vY, size);
  _axpy(ex, size, alpha, gpu_vX, strd, gpu_vY, strd);
  _asum(ex, size, gpu_vY, strd, gpu_vR);
  _dot(ex, size, gpu_vX, strd, gpu_vY, strd, gpu_vS);
  _nrm2(ex, size, gpu_vY, strd, gpu_vT);
  _iamax(ex, size, gpu_vY, strd, gpu_vImax);
  _iamin(ex, size, gpu_vY, strd, gpu_vImin);
  _rot(ex, size, gpu_vX, strd, gpu_vY, strd, _cos, _sin);
  _dot(ex, size, gpu_vX, strd, gpu_vY, strd, gpu_vU);
  _swap(ex, size, gpu_vX, strd, gpu_vY, strd);
  ex.copy_to_host(gpu_vR, vR.data(), 1);
  ex.copy_to_host(gpu_vS, vS.data(), 1);
  ex.copy_to_host(gpu_vT, vT.data(), 1);
  ex.copy_to_host(gpu_vU, vU.data(), 1);
  ex.copy_to_host(gpu_vImax, vImax.data(), 1);
  ex.copy_to_host(gpu_vImin, vImin.data(), 1);
  ex.copy_to_host(gpu_vX, vX.data(), size);
  ex.copy_to_host(gpu_vY, vY.data(), size);

  // because there is a lot of operations, it makes sense to set the precision
  // threshold
  ScalarT prec_sample = std::max(
      std::numeric_limits<ScalarT>::epsilon() * size * 2, prec * ScalarT(1e1));
  // checking that precision is reasonable
  EXPECT_LE(prec_sample, prec * 1e4);
  DEBUG_PRINT(
      std::cout << "prec==" << std::fixed
                << std::setprecision(std::numeric_limits<ScalarT>::digits10)
                << prec_sample << std::endl);
  // compare the results
  EXPECT_NEAR(asum, vR[0], prec_sample);
  EXPECT_NEAR(dot, vS[0], prec_sample);
  EXPECT_NEAR(nrmY, vT[0], prec_sample);
  EXPECT_EQ(imax, vImax[0].getInd());
  EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
  EXPECT_EQ(imin, vImin[0].getInd());
  EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
  EXPECT_NEAR(giv, vU[0], prec_sample);
  EXPECT_NEAR(diff, (vX[0] - vY[0]) + (vX.back() - vY.back()), prec_sample);
  ex.template deallocate<ScalarT>(gpu_vX);
  ex.template deallocate<ScalarT>(gpu_vY);
  ex.template deallocate<ScalarT>(gpu_vR);
  ex.template deallocate<ScalarT>(gpu_vS);
  ex.template deallocate<ScalarT>(gpu_vT);
  ex.template deallocate<ScalarT>(gpu_vU);
  ex.template deallocate<IndVal<ScalarT>>(gpu_vImax);
  ex.template deallocate<IndVal<ScalarT>>(gpu_vImin);
}
