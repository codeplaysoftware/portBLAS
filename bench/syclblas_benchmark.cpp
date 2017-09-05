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

#include <interface/blas1_interface_sycl.hpp>

using namespace blas;

template <typename Device = SYCLDevice>
class SyclBlasBenchmarker {
  cl::sycl::queue q;
  Device dev;

 public:
  SyclBlasBenchmarker():
    q(cl::sycl::default_selector(), [=](cl::sycl::exception_list eL) {
        for (auto &e : eL) {
          try {
            std::rethrow_exception(e);
          } catch (cl::sycl::exception &e) {
            std::cout << " E " << e.what() << std::endl;
          } catch (...) {
            std::cout << " An exception " << std::endl;
          }
        }
    }), dev(q)
  {}

  template <typename ScalarT>
  static cl::sycl::buffer<ScalarT, 1> mkbuffer(ScalarT *data, size_t len) {
    return cl::sycl::buffer<ScalarT, 1>(data, len);
  }

  BENCHMARK_FUNCTION(scal_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT alpha(2.4367453465);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      flops = benchmark<>::measure(no_reps, size * 1, [&]() {
        blas::execute(dev, _scal(size, alpha, buf1, 0, 1));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(axpy_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    ScalarT alpha(2.4367453465);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf2 = mkbuffer<ScalarT>(v2, size);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _axpy(size, alpha, buf1, 0, 1, buf2, 0, 1));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    release_data(v2);
    return flops;
  }

  BENCHMARK_FUNCTION(asum_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT vr;
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto bufR = mkbuffer<ScalarT>(&vr, 1);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _asum(size, buf1, 0, 1, bufR));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(nrm2_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT vr;
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto bufR = mkbuffer<ScalarT>(&vr, 1);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _nrm2(size, buf1, 0, 1, bufR));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(dot_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    ScalarT vr;
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf2 = mkbuffer<ScalarT>(v2, size);
      auto bufR = mkbuffer<ScalarT>(&vr, 1);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _dot(size, buf1, 0, 1, buf2, 0, 1, bufR));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    release_data(v2);
    return flops;
  }

  BENCHMARK_FUNCTION(iamax_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    IndVal<ScalarT> vI(std::numeric_limits<int>::max(), 0);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf_i = mkbuffer<IndVal<ScalarT>>(&vI, 1);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _iamax(size, buf1, 0, 1, buf_i));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(scal2op_bench) {
    using ScalarT = TypeParam;
    ScalarT alpha(2.4367453465);
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT *v2 = new_data<ScalarT>(size);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf2 = mkbuffer<ScalarT>(v2, size);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        blas::execute(dev, _scal(size, alpha, buf1, 0, 1));
        blas::execute(dev, _scal(size, alpha, buf2, 0, 1));
        q.wait_and_throw();
      });
    }
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
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf2 = mkbuffer<ScalarT>(v2, size);
      auto buf3 = mkbuffer<ScalarT>(v3, size);

      flops = benchmark<>::measure(no_reps, size * 3, [&]() {
        blas::execute(dev, _scal(size, alpha, buf1, 0, 1));
        blas::execute(dev, _scal(size, alpha, buf2, 0, 1));
        blas::execute(dev, _scal(size, alpha, buf3, 0, 1));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    release_data(v2);
    release_data(v3);
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
    {
      auto bufsrc1 = mkbuffer<ScalarT>(vsrc1, size);
      auto bufsrc2 = mkbuffer<ScalarT>(vsrc2, size);
      auto bufsrc3 = mkbuffer<ScalarT>(vsrc3, size);
      auto bufdst1 = mkbuffer<ScalarT>(vdst1, size);
      auto bufdst2 = mkbuffer<ScalarT>(vdst2, size);
      auto bufdst3 = mkbuffer<ScalarT>(vdst3, size);

      flops = benchmark<>::measure(no_reps, size * 3 * 2, [&]() {
        blas::execute(dev, _axpy(size, alphas[0], bufsrc1, 0, 1, bufdst1, 0, 1));
        blas::execute(dev, _axpy(size, alphas[1], bufsrc2, 0, 1, bufdst2, 0, 1));
        blas::execute(dev, _axpy(size, alphas[2], bufsrc3, 0, 1, bufdst3, 0, 1));
        q.wait_and_throw();
      });
    }
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
    ScalarT vr[4];
    IndVal<ScalarT> vImax(std::numeric_limits<int>::max(), 0);
    ScalarT alpha(3.135345123);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf2 = mkbuffer<ScalarT>(v2, size);
      auto bufr0 = mkbuffer<ScalarT>(&vr[0], 1);
      auto bufr1 = mkbuffer<ScalarT>(&vr[1], 1);
      auto bufr2 = mkbuffer<ScalarT>(&vr[2], 1);
      auto bufr3 = mkbuffer<ScalarT>(&vr[3], 1);
      auto buf_i1 = mkbuffer<IndVal<ScalarT>>(&vImax, 1);

      flops = benchmark<>::measure(no_reps, size * 12, [&]() {
        blas::execute(dev, _axpy(size, alpha, buf1, 0, 1, buf2, 0, 1));
        blas::execute(dev, _asum(size, buf2, 0, 1, bufr0));
        blas::execute(dev, _dot(size, buf1, 0, 1, buf2, 0, 1, bufr1));
        blas::execute(dev, _nrm2(size, buf2, 0, 1, bufr2));
        blas::execute(dev, _iamax(size, buf2, 0, 1, buf_i1));
        blas::execute(dev, _dot(size, buf1, 0, 1, buf2, 0, 1, bufr3));
        q.wait_and_throw();
      });
    }
    release_data(v1);
    release_data(v2);
    return flops;
  }
};

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 24, 10);
SyclBlasBenchmarker<SYCLDevice> blasbenchmark;

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

BENCHMARK_REGISTER_FUNCTION("iamax_float", iamax_bench<float>);
BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal2op_float", scal2op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal2op_double", scal2op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal3op_float", scal3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal3op_double", scal3op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("axpy3op_float", axpy3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy3op_double", axpy3op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("blas1_float", blas1_bench<float>);
BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench<double>);

BENCHMARK_MAIN_END();
