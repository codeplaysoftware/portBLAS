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
 *  @filename blas1_iface_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<>> BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);
// The register size has been reduced in order to make sure that the output
// result range is within the IEEE 754-1985 standard for floating-point numbers.
REGISTER_SIZE(1023, interface1_test)
REGISTER_STRD(1, interface1_test)
REGISTER_PREC(float, 1e-4, interface1_test)
REGISTER_PREC(double, 1e-6, interface1_test)

TYPED_TEST(BLAS_Test, interface1_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class interface1_test;
  using index_t = int;
  index_t size = TestClass::template test_size<test>();
  index_t strd = TestClass::template test_strd<test>();
  scalar_t prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // creating three random vectors
  std::vector<scalar_t> vX(size);
  std::vector<scalar_t> vY(size);
  std::vector<scalar_t> vZ(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);
  TestClass::set_rand(vZ, size);

  // the values will first be computed in a for loop:
  index_t imax = 0, imin = 0;
  scalar_t asum(0);
  const scalar_t alpha(0.432);
  scalar_t dot(0);
  scalar_t nrmX(0);
  scalar_t nrmY(0);
  scalar_t max(0);
  scalar_t min(std::numeric_limits<scalar_t>::max());
  scalar_t diff(0);
  scalar_t _cos(0);
  scalar_t _sin(0);
  scalar_t giv(0);
  for (int i = 0; i < size; i += strd) {
    scalar_t &x = vX[i];
    scalar_t &y = vY[i];
    scalar_t &z = vZ[i];

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
      scalar_t n1 = x, n2 = z;
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
  std::vector<scalar_t> vR(1);
  // for dot:
  std::vector<scalar_t> vS(1);
  // for nrm2:
  std::vector<scalar_t> vT(1);
  // for dot after _rot
  std::vector<scalar_t> vU(1);
  // for iamax/iamin
  constexpr auto max_val =
      constant<IndexValueTuple<scalar_t, index_t>, const_val::imax>::value();
  std::vector<IndexValueTuple<scalar_t, index_t>> vImax(1, max_val);
  constexpr auto min_val =
      constant<IndexValueTuple<scalar_t, index_t>, const_val::imin>::value();
  std::vector<IndexValueTuple<scalar_t, index_t>> vImin(1, min_val);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vY = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vR = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto gpu_vS = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto gpu_vT = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto gpu_vU = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto gpu_vImax =
      ex.get_policy_handler()
          .template allocate<IndexValueTuple<scalar_t, index_t>>(1);
  auto gpu_vImin =
      ex.get_policy_handler()
          .template allocate<IndexValueTuple<scalar_t, index_t>>(1);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vY.data(), gpu_vY, size);
  _axpy(ex, size, alpha, gpu_vX, strd, gpu_vY, strd);
  _asum(ex, size, gpu_vY, strd, gpu_vR);
  _dot(ex, size, gpu_vX, strd, gpu_vY, strd, gpu_vS);
  _nrm2(ex, size, gpu_vY, strd, gpu_vT);
  _iamax(ex, size, gpu_vY, strd, gpu_vImax);
  _iamin(ex, size, gpu_vY, strd, gpu_vImin);
  _rot(ex, size, gpu_vX, strd, gpu_vY, strd, _cos, _sin);
  _dot(ex, size, gpu_vX, strd, gpu_vY, strd, gpu_vU);
  _swap(ex, size, gpu_vX, strd, gpu_vY, strd);
  auto event0 = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  auto event1 = ex.get_policy_handler().copy_to_host(gpu_vS, vS.data(), 1);
  auto event2 = ex.get_policy_handler().copy_to_host(gpu_vT, vT.data(), 1);
  auto event3 = ex.get_policy_handler().copy_to_host(gpu_vU, vU.data(), 1);
  auto event4 =
      ex.get_policy_handler().copy_to_host(gpu_vImax, vImax.data(), 1);
  auto event5 =
      ex.get_policy_handler().copy_to_host(gpu_vImin, vImin.data(), 1);
  auto event6 = ex.get_policy_handler().copy_to_host(gpu_vX, vX.data(), size);
  auto event7 = ex.get_policy_handler().copy_to_host(gpu_vY, vY.data(), size);
  ex.get_policy_handler().wait(event0, event1, event2, event3, event4, event5,
                               event6, event7);

  // because there is a lot of operations, it makes sense to set the precision
  // threshold
  scalar_t prec_sample =
      std::max(std::numeric_limits<scalar_t>::epsilon() * size * 2,
               prec * scalar_t(1e1));
  // checking that precision is reasonable
  EXPECT_LE(prec_sample, prec * 1e4);
  DEBUG_PRINT(
      std::cout << "prec==" << std::fixed
                << std::setprecision(std::numeric_limits<scalar_t>::digits10)
                << prec_sample << std::endl);
  // compare the results
  EXPECT_NEAR(asum, vR[0], prec_sample);
  EXPECT_NEAR(dot, vS[0], prec_sample);
  EXPECT_NEAR(nrmY, vT[0], prec_sample);
  EXPECT_EQ(imax, vImax[0].get_index());
  EXPECT_NEAR(max, vImax[0].get_value(), prec_sample);
  EXPECT_EQ(imin, vImin[0].get_index());
  EXPECT_NEAR(max, vImax[0].get_value(), prec_sample);
  EXPECT_NEAR(giv, vU[0], prec_sample);
  EXPECT_NEAR(diff, (vX[0] - vY[0]) + (vX.back() - vY.back()), prec_sample);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vX);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vY);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vR);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vS);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vT);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vU);
  ex.get_policy_handler()
      .template deallocate<IndexValueTuple<scalar_t, index_t>>(gpu_vImax);
  ex.get_policy_handler()
      .template deallocate<IndexValueTuple<scalar_t, index_t>>(gpu_vImin);
}
