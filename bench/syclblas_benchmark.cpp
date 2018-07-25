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

#include "blas_benchmark.hpp"

#include <sycl-blas/interface/blas1_interface.hpp>

using namespace blas;

template <typename ExecutorType = SYCL>
class SyclBlasBenchmarker {
  cl::sycl::queue q;
  Executor<ExecutorType> ex;

 public:
  SyclBlasBenchmarker()
      : q(cl::sycl::default_selector(),
          [=](cl::sycl::exception_list eL) {
            for (auto &e : eL) {
              try {
                std::rethrow_exception(e);
              } catch (cl::sycl::exception &e) {
                std::cout << " E " << e.what() << std::endl;
              } catch (...) {
                std::cout << " An exception " << std::endl;
              }
            }
          }),
        ex(q) {}

  BENCHMARK_FUNCTION(scal_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT alpha(2.4367453465);
    double flops;
    auto in = ex.template allocate<ScalarT>(size);
    ex.copy_to_device(v1, in, size);
    flops = benchmark<>::measure(no_reps, size * 1, [&]() {
      _scal(ex, size, alpha, in, 1);
      ex.sycl_queue().wait_and_throw();
    });
    ex.template deallocate<ScalarT>(in);
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(axpy_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    ScalarT alpha(2.4367453465);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto iny = ex.template allocate<ScalarT>(size);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(v2, iny, size);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _axpy(ex, size, alpha, inx, 1, iny, 1);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(iny);
    release_data(v1);
    release_data(v2);
    return flops;
  }

  BENCHMARK_FUNCTION(asum_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT vr;
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto inr = ex.template allocate<ScalarT>(1);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(&vr, inr, 1);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _asum(ex, size, inx, 1, inr);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(inr);
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(nrm2_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto inr = ex.template allocate<ScalarT>(1);
    ex.copy_to_device(v1, inx, size);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _nrm2(ex, size, inx, 1, inr);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(inr);
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(dot_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto iny = ex.template allocate<ScalarT>(size);
    auto inr = ex.template allocate<ScalarT>(1);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(v2, iny, size);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _dot(ex, size, inx, 1, iny, 1, inr);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(iny);
    ex.template deallocate<ScalarT>(inr);
    release_data(v1);
    release_data(v2);
    return flops;
  }

  BENCHMARK_FUNCTION(iamax_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto outI = ex.template allocate<IndexValueTuple<ScalarT>>(1);
    ex.copy_to_device(v1, inx, size);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _iamax(ex, size, inx, 1, outI);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<IndexValueTuple<ScalarT>>(outI);
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(iamin_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    auto inx = ex.template allocate<ScalarT>(size);
    auto outI = ex.template allocate<IndexValueTuple<ScalarT>>(1);
    ex.copy_to_device(v1, inx, size);
    double flops;

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _iamin(ex, size, inx, 1, outI);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<IndexValueTuple<ScalarT>>(outI);
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(scal2op_bench) {
    using ScalarT = TypeParam;
    ScalarT alpha(2.4367453465);
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    double flops;

    auto inx = ex.template allocate<ScalarT>(size);
    auto iny = ex.template allocate<ScalarT>(size);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(v2, iny, size);

    flops = benchmark<>::measure(no_reps, size * 2, [&]() {
      _scal(ex, size, alpha, inx, 1);
      _scal(ex, size, alpha, iny, 1);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(iny);
    release_data(v1);
    release_data(v2);
    return flops;
  }

  BENCHMARK_FUNCTION(scal3op_bench) {
    using ScalarT = TypeParam;
    ScalarT alpha(2.4367453465);
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    ScalarT *v3 = new_data<ScalarT>(size);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto iny = ex.template allocate<ScalarT>(size);
    auto inz = ex.template allocate<ScalarT>(size);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(v2, iny, size);
    ex.copy_to_device(v3, inz, size);

    flops = benchmark<>::measure(no_reps, size * 3, [&]() {
      _scal(ex, size, alpha, inx, 1);
      _scal(ex, size, alpha, iny, 1);
      _scal(ex, size, alpha, inz, 1);
      ex.sycl_queue().wait_and_throw();
    });

    release_data(v1);
    release_data(v2);
    release_data(v3);
    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(iny);
    ex.template deallocate<ScalarT>(inz);
    return flops;
  }

  BENCHMARK_FUNCTION(axpy3op_bench) {
    using ScalarT = TypeParam;
    std::array<ScalarT, 3> alphas = {1.78426458744, 2.187346575843,
                                     3.78164387328};
    ScalarT *vsrc1 = new_data<ScalarT>(size);
    ScalarT *vsrc2 = new_data<ScalarT>(size);
    ScalarT *vsrc3 = new_data<ScalarT>(size);
    ScalarT *vdst1 = new_data<ScalarT>(size);
    ScalarT *vdst2 = new_data<ScalarT>(size);
    ScalarT *vdst3 = new_data<ScalarT>(size);
    double flops;

    auto insrc1 = ex.template allocate<ScalarT>(size);
    auto indst1 = ex.template allocate<ScalarT>(size);
    auto insrc2 = ex.template allocate<ScalarT>(size);
    auto indst2 = ex.template allocate<ScalarT>(size);
    auto insrc3 = ex.template allocate<ScalarT>(size);
    auto indst3 = ex.template allocate<ScalarT>(size);
    ex.copy_to_device(vsrc1, insrc1, size);
    ex.copy_to_device(vdst1, indst1, size);
    ex.copy_to_device(vsrc2, insrc2, size);
    ex.copy_to_device(vdst2, indst2, size);
    ex.copy_to_device(vsrc3, insrc3, size);
    ex.copy_to_device(vdst3, indst3, size);

    flops = benchmark<>::measure(no_reps, size * 3 * 2, [&]() {
      _axpy(ex, size, alphas[0], insrc1, 1, indst1, 1);
      _axpy(ex, size, alphas[1], insrc2, 1, indst2, 1);
      _axpy(ex, size, alphas[2], insrc3, 1, indst3, 1);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(insrc1);
    ex.template deallocate<ScalarT>(indst1);
    ex.template deallocate<ScalarT>(insrc2);
    ex.template deallocate<ScalarT>(indst2);
    ex.template deallocate<ScalarT>(insrc3);
    ex.template deallocate<ScalarT>(indst3);
    release_data(vsrc1);
    release_data(vsrc2);
    release_data(vsrc3);
    release_data(vdst1);
    release_data(vdst2);
    release_data(vdst3);
    return flops;
  }

  BENCHMARK_FUNCTION(blas1_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    ScalarT alpha(3.135345123);
    double flops;
    auto inx = ex.template allocate<ScalarT>(size);
    auto iny = ex.template allocate<ScalarT>(size);
    auto inr1 = ex.template allocate<ScalarT>(1);
    auto inr2 = ex.template allocate<ScalarT>(1);
    auto inr3 = ex.template allocate<ScalarT>(1);
    auto inr4 = ex.template allocate<ScalarT>(1);
    auto inrI = ex.template allocate<IndexValueTuple<ScalarT>>(1);
    ex.copy_to_device(v1, inx, size);
    ex.copy_to_device(v2, iny, size);

    flops = benchmark<>::measure(no_reps, size * 12, [&]() {
      _axpy(ex, size, alpha, inx, 1, iny, 1);
      _asum(ex, size, iny, 1, inr1);
      _dot(ex, size, inx, 1, iny, 1, inr2);
      _nrm2(ex, size, iny, 1, inr3);
      _iamax(ex, size, iny, 1, inrI);
      _dot(ex, size, inx, 1, iny, 1, inr4);
      ex.sycl_queue().wait_and_throw();
    });

    ex.template deallocate<ScalarT>(inx);
    ex.template deallocate<ScalarT>(iny);
    ex.template deallocate<ScalarT>(inr1);
    ex.template deallocate<ScalarT>(inr2);
    ex.template deallocate<ScalarT>(inr3);
    ex.template deallocate<ScalarT>(inr4);
    ex.template deallocate<IndexValueTuple<ScalarT>>(inrI);
    release_data(v1);
    release_data(v2);
    return flops;
  }
};

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 24, 10);
SyclBlasBenchmarker<SYCL> blasbenchmark;

BENCHMARK_REGISTER_FUNCTION("scal_float", scal_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal_double", scal_bench<double>);

BENCHMARK_REGISTER_FUNCTION("axpy_float", axpy_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy_double", axpy_bench<double>);

BENCHMARK_REGISTER_FUNCTION("asum_float", asum_bench<float>);
BENCHMARK_REGISTER_FUNCTION("asum_double", asum_bench<double>);

BENCHMARK_REGISTER_FUNCTION("nrm2_float", nrm2_bench<float>);
BENCHMARK_REGISTER_FUNCTION("nrm2_double", nrm2_bench<double>);

BENCHMARK_REGISTER_FUNCTION("dot_float", dot_bench<float>);
BENCHMARK_REGISTER_FUNCTION("dot_double", dot_bench<double>);

BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal2op_float", scal2op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal2op_double", scal2op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal3op_float", scal3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal3op_double", scal3op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("axpy3op_float", axpy3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy3op_double", axpy3op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench<double>);

BENCHMARK_MAIN_END();
