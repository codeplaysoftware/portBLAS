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

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double>>
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, interface1_test)
REGISTER_STRD(1, interface1_test)
REGISTER_PREC(float, 1e-4, interface1_test)
REGISTER_PREC(double, 1e-6, interface1_test)

B1_TEST(interface1_test) {
  UNPACK_PARAM(interface1_test)
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  std::cout << "size == " << size << std::endl;
  std::vector<ScalarT> vX_(size);
  std::vector<ScalarT> vY_(size);
  TestClass::set_rand(vX_, size);
  TestClass::set_rand(vY_, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    auto vX = vX_;
    auto vY = vY_;
    std::vector<ScalarT> vZ(size);

    std::vector<ScalarT> vR(1);
    std::vector<ScalarT> vS(1);
    std::vector<ScalarT> vT(1);
    std::vector<ScalarT> vU(1);

    std::vector<IndVal<ScalarT>> vImax(
        1, constant<IndVal<ScalarT>, const_val::imax>::value);
    std::vector<IndVal<ScalarT>> vImin(
        1, constant<IndVal<ScalarT>, const_val::imin>::value);
    size_t imax = 0, imin = 0;
    ScalarT asum(0);
    ScalarT alpha(0.0);
    ScalarT dot(0);
    ScalarT nrmX(0);
    ScalarT nrmY(0);
    ScalarT max(0);
    ScalarT min(std::numeric_limits<ScalarT>::max());
    ScalarT diff(0);
    ScalarT _cos(0);
    ScalarT _sin(0);
    ScalarT giv(0);
    for (size_t i = 0; i < size; i += i) {
      ScalarT &x = vX[i];
      ScalarT &y = vY[i];
      ScalarT &z = vZ[i];

      z = x * alpha + y;
      asum += std::abs(z);
      dot += x * z;
      nrmX += x * x, nrmY += z * z;
      if (std::abs(z) > std::abs(max)) {
        max = z;
        imax = i;
      }
      if (std::abs(z) < std::abs(min)) {
        min = z;
        imin = i;
      }
      if (i == 0) {
        ScalarT n1 = x;
        ScalarT n2 = z;
        _rotg(n1, n2, _cos, _sin);
        diff = (z * _cos - x * _sin) - (x * _cos + z * _sin);
      } else if (i == size - 1) {
        diff += (z * _cos - x * _sin) - (x * _cos + z * _sin);
      }
      giv += ((x * _cos + z * _sin) * (z * _cos - x * _sin));
    }
    nrmX = std::sqrt(nrmX), nrmY = std::sqrt(nrmY);

    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vY = TestClass::make_buffer(vY);
      auto buf_vR = TestClass::make_buffer(vR);
      auto buf_vS = TestClass::make_buffer(vS);
      auto buf_vT = TestClass::make_buffer(vT);
      auto buf_vU = TestClass::make_buffer(vU);
      auto buf_vImax = TestClass::make_buffer(vImax);
      auto buf_vImin = TestClass::make_buffer(vImin);

      _axpy(ex, size, alpha, buf_vX, strd, buf_vY, strd);
      _asum(ex, size, buf_vY, strd, buf_vR);
      _dot(ex, size, buf_vX, strd, buf_vY, strd, buf_vS);
      _nrm2(ex, size, buf_vY, strd, buf_vT);
      _iamax(ex, size, buf_vY, strd, buf_vImax);
      _iamin(ex, size, buf_vY, strd, buf_vImin);
      _rot(ex, size, buf_vX, strd, buf_vY, strd, _cos, _sin);
      _dot(ex, size, buf_vX, strd, buf_vY, strd, buf_vU);
      _swap(ex, size, buf_vX, strd, buf_vY, strd);
    }
    ScalarT prec_sample =
        std::max(std::numeric_limits<ScalarT>::epsilon() * size * 2,
                 prec * ScalarT(1e1));
    EXPECT_LE(prec_sample, prec * 1e4);
    std::cout << "prec==" << std::fixed
              << std::setprecision(std::numeric_limits<ScalarT>::digits10)
              << prec_sample << std::endl;
    EXPECT_NEAR(asum, vR[0], prec_sample);
    EXPECT_NEAR(dot, vS[0], prec_sample);
    EXPECT_NEAR(nrmY, vT[0], prec_sample);
    EXPECT_EQ(imax, vImax[0].getInd());
    EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
    EXPECT_EQ(imin, vImin[0].getInd());
    EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
    /* EXPECT_NEAR(giv, vU[0], prec_sample); */
    EXPECT_NEAR(diff, (vX[0] - vY[0]) + (vX.back() - vY.back()), prec_sample);
  }
}
