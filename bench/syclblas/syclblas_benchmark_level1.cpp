/**********************
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

#include "sycl_blas.h"

using namespace blas;

BENCHMARK_NAME_FORMAT(syclblas_level_1) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_" << params;
  return fname.str();
}

BENCHMARK(scal, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  auto in = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(v1.data(), in, size);
  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 1, [&]() -> std::vector<cl::sycl::event> {
        auto event = _scal(ex, size, alpha, in, 1);
        ex.get_policy_handler().wait(event);
        return event;
      });
  ex.get_policy_handler().template deallocate<scalar_t>(in);
  return flops;
}

BENCHMARK(axpy, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark<>::random_data<scalar_t>(size);
  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto iny = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(v2.data(), iny, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _axpy(ex, size, alpha, inx, 1, iny, 1);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(iny);
  return flops;
}

BENCHMARK(asum, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  scalar_t vr;

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto inr = ex.get_policy_handler().template allocate<scalar_t>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(&vr, inr, 1);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _asum(ex, size, inx, 1, inr);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(inr);
  return flops;
}

BENCHMARK(nrm2, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto inr = ex.get_policy_handler().template allocate<scalar_t>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _nrm2(ex, size, inx, 1, inr);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(inr);
  return flops;
}

BENCHMARK(dot, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark<>::random_data<scalar_t>(size);

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto iny = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto inr = ex.get_policy_handler().template allocate<scalar_t>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(v2.data(), iny, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _dot(ex, size, inx, 1, iny, 1, inr);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(iny);
  ex.get_policy_handler().template deallocate<scalar_t>(inr);
  return flops;
}

BENCHMARK(iamax, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto outI = ex.get_policy_handler()
                  .template allocate<Indexvalue_tuple<scalar_t, index_t>>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _iamax(ex, size, inx, 1, outI);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler()
      .template deallocate<Indexvalue_tuple<scalar_t, index_t>>(outI);
  return flops;
}

BENCHMARK(iamin, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto outI = ex.get_policy_handler()
                  .template allocate<Indexvalue_tuple<scalar_t, index_t>>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event = _iamin(ex, size, inx, 1, outI);
        ex.get_policy_handler().wait(event);
        return event;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler()
      .template deallocate<Indexvalue_tuple<scalar_t, index_t>>(outI);
  return flops;
}

BENCHMARK(scal2op, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();
  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark<>::random_data<scalar_t>(size);

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto iny = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(v2.data(), iny, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event0 = _scal(ex, size, alpha, inx, 1);
        auto event1 = _scal(ex, size, alpha, iny, 1);
        ex.get_policy_handler().wait(event0, event1);
        return event1;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(iny);
  return flops;
}

BENCHMARK(scal3op, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();
  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v3 = benchmark<>::random_data<scalar_t>(size);

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto iny = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto inz = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(v2.data(), iny, size);
  ex.get_policy_handler().copy_to_device(v3.data(), inz, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 3, [&]() -> std::vector<cl::sycl::event> {
        auto event0 = _scal(ex, size, alpha, inx, 1);
        auto event1 = _scal(ex, size, alpha, iny, 1);
        auto event2 = _scal(ex, size, alpha, inz, 1);
        ex.get_policy_handler().wait(event0, event1, event2);
        return event2;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(iny);
  ex.get_policy_handler().template deallocate<scalar_t>(inz);
  return flops;
}

BENCHMARK(axpy3op, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::array<scalar_t, 3> alphas = {1.78426458744, 2.187346575843,
                                   3.78164387328};
  std::vector<scalar_t> vsrc1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> vsrc2 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> vsrc3 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst2 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst3 = benchmark<>::random_data<scalar_t>(size);

  auto insrc1 = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto indst1 = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto insrc2 = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto indst2 = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto insrc3 = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto indst3 = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(vsrc1.data(), insrc1, size);
  ex.get_policy_handler().copy_to_device(vdst1.data(), indst1, size);
  ex.get_policy_handler().copy_to_device(vsrc2.data(), insrc2, size);
  ex.get_policy_handler().copy_to_device(vdst2.data(), indst2, size);
  ex.get_policy_handler().copy_to_device(vsrc3.data(), insrc3, size);
  ex.get_policy_handler().copy_to_device(vdst3.data(), indst3, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 3 * 2, [&]() -> std::vector<cl::sycl::event> {
        auto event0 = _axpy(ex, size, alphas[0], insrc1, 1, indst1, 1);
        auto event1 = _axpy(ex, size, alphas[1], insrc2, 1, indst2, 1);
        auto event2 = _axpy(ex, size, alphas[2], insrc3, 1, indst3, 1);
        ex.get_policy_handler().wait(event0, event1, event2);
        return event2;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(insrc1);
  ex.get_policy_handler().template deallocate<scalar_t>(indst1);
  ex.get_policy_handler().template deallocate<scalar_t>(insrc2);
  ex.get_policy_handler().template deallocate<scalar_t>(indst2);
  ex.get_policy_handler().template deallocate<scalar_t>(insrc3);
  ex.get_policy_handler().template deallocate<scalar_t>(indst3);
  return flops;
}

BENCHMARK(blas1, syclblas_level_1) {
  using scalar_t = ElemT;
  using index_t = int;
  const index_t size = params;

  std::vector<scalar_t> v1 = benchmark<>::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark<>::random_data<scalar_t>(size);
  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  auto inx = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto iny = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto inr1 = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto inr2 = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto inr3 = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto inr4 = ex.get_policy_handler().template allocate<scalar_t>(1);
  auto inrI = ex.get_policy_handler()
                  .template allocate<Indexvalue_tuple<scalar_t, index_t>>(1);
  ex.get_policy_handler().copy_to_device(v1.data(), inx, size);
  ex.get_policy_handler().copy_to_device(v2.data(), iny, size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, size * 12, [&]() -> std::vector<cl::sycl::event> {
        auto event0 = _axpy(ex, size, alpha, inx, 1, iny, 1);
        auto event1 = _asum(ex, size, iny, 1, inr1);
        auto event2 = _dot(ex, size, inx, 1, iny, 1, inr2);
        auto event3 = _nrm2(ex, size, iny, 1, inr3);
        auto event4 = _iamax(ex, size, iny, 1, inrI);
        auto event5 = _dot(ex, size, inx, 1, iny, 1, inr4);
        ex.get_policy_handler().wait(event0, event1, event2, event3, event4,
                                     event5);
        return event5;
      });

  ex.get_policy_handler().template deallocate<scalar_t>(inx);
  ex.get_policy_handler().template deallocate<scalar_t>(iny);
  ex.get_policy_handler().template deallocate<scalar_t>(inr1);
  ex.get_policy_handler().template deallocate<scalar_t>(inr2);
  ex.get_policy_handler().template deallocate<scalar_t>(inr3);
  ex.get_policy_handler().template deallocate<scalar_t>(inr4);
  ex.get_policy_handler()
      .template deallocate<Indexvalue_tuple<scalar_t, index_t>>(inrI);
  return flops;
}

SUITE(ADD(scal), ADD(axpy), ADD(asum), ADD(nrm2), ADD(dot), ADD(scal2op),
      ADD(iamax), ADD(scal3op), ADD(axpy3op), ADD(blas1))

SYCL_BENCHMARK_MAIN(default_ranges::level_1, 10);
