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
 *  @filename syclblas_benchmark.cpp
 *
 **************************************************************************/

#include "../common/blas_benchmark.hpp"

#include <interface/blas1_interface.hpp>

using namespace blas;

BENCHMARK_NAME_FORMAT(syclblas_level_1) {
  std::ostringstream fname;
  fname << typeid(ElemT).name() << "_" << name() << "_" << params;
  return fname.str();
}

BENCHMARK(scal, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  ScalarT alpha(2.4367453465);

  auto in = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(v1.data(), in, size);
  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 1, [&]() {
        auto event = _scal(ex, size, alpha, in, 1);
        ex.wait(event);
      });
  ex.template deallocate<ScalarT>(in);
  return flops;
}

BENCHMARK(axpy, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark<>::random_data<ScalarT>(size);
  ScalarT alpha(2.4367453465);

  auto inx = ex.template allocate<ScalarT>(size);
  auto iny = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(v2.data(), iny, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _axpy(ex, size, alpha, inx, 1, iny, 1);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(iny);
  return flops;
}

BENCHMARK(asum, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  ScalarT vr;

  auto inx = ex.template allocate<ScalarT>(size);
  auto inr = ex.template allocate<ScalarT>(1);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(&vr, inr, 1);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _asum(ex, size, inx, 1, inr);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(inr);
  return flops;
}

BENCHMARK(nrm2, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);

  auto inx = ex.template allocate<ScalarT>(size);
  auto inr = ex.template allocate<ScalarT>(1);
  ex.copy_to_device(v1.data(), inx, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _nrm2(ex, size, inx, 1, inr);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(inr);
  return flops;
}

BENCHMARK(dot, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;

  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark<>::random_data<ScalarT>(size);

  auto inx = ex.template allocate<ScalarT>(size);
  auto iny = ex.template allocate<ScalarT>(size);
  auto inr = ex.template allocate<ScalarT>(1);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(v2.data(), iny, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _dot(ex, size, inx, 1, iny, 1, inr);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(iny);
  ex.template deallocate<ScalarT>(inr);
  return flops;
}

BENCHMARK(iamax, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);

  auto inx = ex.template allocate<ScalarT>(size);
  auto outI = ex.template allocate<IndexValueTuple<ScalarT, IndexType>>(1);
  ex.copy_to_device(v1.data(), inx, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _iamax(ex, size, inx, 1, outI);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<IndexValueTuple<ScalarT, IndexType>>(outI);
  return flops;
}

BENCHMARK(iamin, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  auto inx = ex.template allocate<ScalarT>(size);
  auto outI = ex.template allocate<IndexValueTuple<ScalarT, IndexType>>(1);
  ex.copy_to_device(v1.data(), inx, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event = _iamin(ex, size, inx, 1, outI);
        ex.wait(event);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<IndexValueTuple<ScalarT, IndexType>>(outI);
  return flops;
}

BENCHMARK(scal2op, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  ScalarT alpha(2.4367453465);
  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark<>::random_data<ScalarT>(size);

  auto inx = ex.template allocate<ScalarT>(size);
  auto iny = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(v2.data(), iny, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 2, [&]() {
        auto event0 = _scal(ex, size, alpha, inx, 1);
        auto event1 = _scal(ex, size, alpha, iny, 1);
        ex.wait(event0, event1);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(iny);
  return flops;
}

BENCHMARK(scal3op, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  ScalarT alpha(2.4367453465);
  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v3 = benchmark<>::random_data<ScalarT>(size);

  auto inx = ex.template allocate<ScalarT>(size);
  auto iny = ex.template allocate<ScalarT>(size);
  auto inz = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(v2.data(), iny, size);
  ex.copy_to_device(v3.data(), inz, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 3, [&]() {
        auto event0 = _scal(ex, size, alpha, inx, 1);
        auto event1 = _scal(ex, size, alpha, iny, 1);
        auto event2 = _scal(ex, size, alpha, inz, 1);
        ex.wait(event0, event1, event2);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(iny);
  ex.template deallocate<ScalarT>(inz);
  return flops;
}

BENCHMARK(axpy3op, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::array<ScalarT, 3> alphas = {1.78426458744, 2.187346575843,
                                   3.78164387328};
  std::vector<ScalarT> vsrc1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> vsrc2 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> vsrc3 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> vdst1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> vdst2 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> vdst3 = benchmark<>::random_data<ScalarT>(size);

  auto insrc1 = ex.template allocate<ScalarT>(size);
  auto indst1 = ex.template allocate<ScalarT>(size);
  auto insrc2 = ex.template allocate<ScalarT>(size);
  auto indst2 = ex.template allocate<ScalarT>(size);
  auto insrc3 = ex.template allocate<ScalarT>(size);
  auto indst3 = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(vsrc1.data(), insrc1, size);
  ex.copy_to_device(vdst1.data(), indst1, size);
  ex.copy_to_device(vsrc2.data(), insrc2, size);
  ex.copy_to_device(vdst2.data(), indst2, size);
  ex.copy_to_device(vsrc3.data(), insrc3, size);
  ex.copy_to_device(vdst3.data(), indst3, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 3 * 2, [&]() {
        auto event0 = _axpy(ex, size, alphas[0], insrc1, 1, indst1, 1);
        auto event1 = _axpy(ex, size, alphas[1], insrc2, 1, indst2, 1);
        auto event2 = _axpy(ex, size, alphas[2], insrc3, 1, indst3, 1);
        ex.wait(event0, event1, event2);
      });

  ex.template deallocate<ScalarT>(insrc1);
  ex.template deallocate<ScalarT>(indst1);
  ex.template deallocate<ScalarT>(insrc2);
  ex.template deallocate<ScalarT>(indst2);
  ex.template deallocate<ScalarT>(insrc3);
  ex.template deallocate<ScalarT>(indst3);
  return flops;
}

BENCHMARK(blas1, syclblas_level_1) {
  using ScalarT = ElemT;
  using IndexType = int;
  IndexType size = params;

  std::vector<ScalarT> v1 = benchmark<>::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark<>::random_data<ScalarT>(size);
  ScalarT alpha(3.135345123);

  auto inx = ex.template allocate<ScalarT>(size);
  auto iny = ex.template allocate<ScalarT>(size);
  auto inr1 = ex.template allocate<ScalarT>(1);
  auto inr2 = ex.template allocate<ScalarT>(1);
  auto inr3 = ex.template allocate<ScalarT>(1);
  auto inr4 = ex.template allocate<ScalarT>(1);
  auto inrI = ex.template allocate<IndexValueTuple<ScalarT, IndexType>>(1);
  ex.copy_to_device(v1.data(), inx, size);
  ex.copy_to_device(v2.data(), iny, size);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, size * 12, [&]() {
        auto event0 = _axpy(ex, size, alpha, inx, 1, iny, 1);
        auto event1 = _asum(ex, size, iny, 1, inr1);
        auto event2 = _dot(ex, size, inx, 1, iny, 1, inr2);
        auto event3 = _nrm2(ex, size, iny, 1, inr3);
        auto event4 = _iamax(ex, size, iny, 1, inrI);
        auto event5 = _dot(ex, size, inx, 1, iny, 1, inr4);
        ex.wait(event0, event1, event2, event3, event4, event5);
      });

  ex.template deallocate<ScalarT>(inx);
  ex.template deallocate<ScalarT>(iny);
  ex.template deallocate<ScalarT>(inr1);
  ex.template deallocate<ScalarT>(inr2);
  ex.template deallocate<ScalarT>(inr3);
  ex.template deallocate<ScalarT>(inr4);
  ex.template deallocate<IndexValueTuple<ScalarT, IndexType>>(inrI);
  return flops;
}

SUITE(ADD(scal), ADD(axpy), ADD(asum), ADD(nrm2), ADD(dot), ADD(scal2op),
      ADD(iamax), ADD(scal3op), ADD(axpy3op), ADD(blas1))

SYCL_BENCHMARK_MAIN(default_ranges::level_1, 10);
