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

  template <typename ScalarT>
  static cl::sycl::buffer<ScalarT, 1> mkbuffer(ScalarT *data, size_t len) {
    return cl::sycl::buffer<ScalarT, 1>(data, len);
  }

  template <typename ScalarT>
  static vector_view<ScalarT, cl::sycl::buffer<ScalarT>> mkvview(
      cl::sycl::buffer<ScalarT, 1> &buf) {
    return vector_view<ScalarT, cl::sycl::buffer<ScalarT>>(buf);
  }

  BENCHMARK_FUNCTION(scal_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    ScalarT alpha(2.4367453465);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto vvw1 = mkvview(buf1);
      flops = benchmark<>::measure(no_reps, size * 1, [&]() {
        _scal(ex, size, alpha, vvw1, 1);
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
      auto vvw1 = mkvview(buf1);
      auto vvw2 = mkvview(buf2);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _axpy(ex, size, alpha, vvw1, 1, vvw2, 1);
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
      auto vvw1 = mkvview(buf1);
      auto vvwR = mkvview(bufR);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _asum(ex, size, vvw1, 1, vvwR);
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
      auto vvw1 = mkvview(buf1);
      auto vvwR = mkvview(bufR);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _nrm2(ex, size, vvw1, 1, vvwR);
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
      auto vvw1 = mkvview(buf1);
      auto vvw2 = mkvview(buf2);
      auto vvwR = mkvview(bufR);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _dot(ex, size, vvw1, 1, vvw2, 1, vvwR);
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
    IndVal<ScalarT> vI(std::numeric_limits<size_t>::max(), 0);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf_i = mkbuffer<IndVal<ScalarT>>(&vI, 1);
      auto vvw1 = mkvview(buf1);
      auto vvw_i = mkvview(buf_i);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _iamax(ex, size, vvw1, 1, vvw_i);
        q.wait_and_throw();
      });
    }
    release_data(v1);
    return flops;
  }

  BENCHMARK_FUNCTION(iamin_bench) {
    using ScalarT = TypeParam;
    ScalarT *v1 = new_data<ScalarT>(size);
    IndVal<ScalarT> vI(std::numeric_limits<size_t>::max(), 0);
    double flops;
    {
      auto buf1 = mkbuffer<ScalarT>(v1, size);
      auto buf_i = mkbuffer<IndVal<ScalarT>>(&vI, 1);
      auto vvw1 = mkvview(buf1);
      auto vvw_i = mkvview(buf_i);
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _iamin(ex, size, vvw1, 1, vvw_i);
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

      auto vvw1 = mkvview(buf1);
      auto vvw2 = mkvview(buf2);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        _scal(ex, size, alpha, vvw1, 1);
        _scal(ex, size, alpha, vvw2, 1);
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

      auto vvw1 = mkvview(buf1);
      auto vvw2 = mkvview(buf2);
      auto vvw3 = mkvview(buf3);

      flops = benchmark<>::measure(no_reps, size * 3, [&]() {
        _scal(ex, size, alpha, vvw1, 1);
        _scal(ex, size, alpha, vvw2, 1);
        _scal(ex, size, alpha, vvw3, 1);
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

      auto vvwsrc1 = mkvview(bufsrc1);
      auto vvwsrc2 = mkvview(bufsrc2);
      auto vvwsrc3 = mkvview(bufsrc3);
      auto vvwdst1 = mkvview(bufdst1);
      auto vvwdst2 = mkvview(bufdst2);
      auto vvwdst3 = mkvview(bufdst3);

      flops = benchmark<>::measure(no_reps, size * 3 * 2, [&]() {
        _axpy(ex, size, alphas[0], vvwsrc1, 1, vvwdst1, 1);
        _axpy(ex, size, alphas[1], vvwsrc2, 1, vvwdst2, 1);
        _axpy(ex, size, alphas[2], vvwsrc3, 1, vvwdst3, 1);
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
    IndVal<ScalarT> vImax(std::numeric_limits<size_t>::max(), 0);
    /* IndVal<ScalarT> vImin(std::numeric_limits<size_t>::max(), 0); */
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
      /* auto buf_i2 = mkbuffer<IndVal<ScalarT>>(&vImin, 1); */

      auto vvw1 = mkvview(buf1);
      auto vvw2 = mkvview(buf2);
      auto vvwr0 = mkvview(bufr0);
      auto vvwr1 = mkvview(bufr1);
      auto vvwr2 = mkvview(bufr2);
      auto vvwr3 = mkvview(bufr3);
      auto vvw_i1 = mkvview(buf_i1);
      /* auto vvw_i2 = mkvview(buf_i2); */

      flops = benchmark<>::measure(no_reps, size * 12, [&]() {
        _axpy(ex, size, alpha, vvw1, 1, vvw2, 1);
        _asum(ex, size, vvw2, 1, vvwr0);
        _dot(ex, size, vvw1, 1, vvw2, 1, vvwr1);
        _nrm2(ex, size, vvw2, 1, vvwr2);
        _iamax(ex, size, vvw2, 1, vvw_i1);
        /* _iamin(ex, size, vvw2, 1, vvw_i2); */
        _dot(ex, size, vvw1, 1, vvw2, 1, vvwr3);
        q.wait_and_throw();
      });
    }
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

// constant<float, imax> is not defined, so the float version will fail
/* BENCHMARK_REGISTER_FUNCTION("iamax_float", iamax_bench<float>); */
BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal2op_float", scal2op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal2op_double", scal2op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("scal3op_float", scal3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal3op_double", scal3op_bench<double>);

BENCHMARK_REGISTER_FUNCTION("axpy3op_float", axpy3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy3op_double", axpy3op_bench<double>);

// constant<float, imax> is not defined, so the float version will fail
/* BENCHMARK_REGISTER_FUNCTION("blas1_float", blas1_bench<float>); */
BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench<double>);

BENCHMARK_MAIN_END();
