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
 *  @filename clblas_benchmark.cpp
 *
 **************************************************************************/

#include "blas_benchmark.hpp"

#include <complex>
#include <vector>

#include <clBLAS.h>

#include "clwrap.hpp"

#define CLBLAS_FUNCTION(postfix)                          \
  template <typename T>                                   \
  struct clblasX##postfix;                                \
  template <>                                             \
  struct clblasX##postfix<float> {                        \
    static constexpr const auto func = &clblasS##postfix; \
  };                                                      \
  template <>                                             \
  struct clblasX##postfix<double> {                       \
    static constexpr const auto func = &clblasD##postfix; \
  };

#define CLBLASI_FUNCTION(postfix)                          \
  template <typename T>                                    \
  struct clblasiX##postfix;                                \
  template <>                                              \
  struct clblasiX##postfix<float> {                        \
    static constexpr const auto func = &clblasiS##postfix; \
  };                                                       \
  template <>                                              \
  struct clblasiX##postfix<double> {                       \
    static constexpr const auto func = &clblasiD##postfix; \
  };
/* template <> struct clblasiX##postfix<std::complex<float>> { \ */
/*   static constexpr const auto value = clblasiC##postfix; \ */
/* }; \ */
/* template <> struct clblasiX##postfix<std::complex<double>> { \ */
/*   static constexpr const auto value = clblasiZ##postfix; \ */
/* }; */

CLBLAS_FUNCTION(scal);
CLBLAS_FUNCTION(asum);
CLBLAS_FUNCTION(axpy);
CLBLAS_FUNCTION(nrm2);
CLBLAS_FUNCTION(dot);
CLBLASI_FUNCTION(amax);

class ClBlasBenchmarker {
  Context context;

 public:
  ClBlasBenchmarker() : context() { clblasSetup(); }

  BENCHMARK_FUNCTION(scal_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(context, size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 1, [&]() {
        clblasXscal<ScalarT>::func(size, alpha, buf1.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(axpy_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(context, size);
      MemBuffer<ScalarT> buf2(context, size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 1, [&]() {
        clblasXaxpy<ScalarT>::func(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0,
                                   1, 1, context._queue(), 0, NULL,
                                   &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(asum_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> buf1(context, size);
      MemBuffer<ScalarT, CL_MEM_READ_ONLY> bufr(context, &vr, 1);
      MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS> scratch(context, size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblasXasum<ScalarT>::func(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                                   scratch.dev(), 1, context._queue(), 0, NULL,
                                   &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(nrm2_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> buf1(context, size);
      MemBuffer<ScalarT, CL_MEM_READ_ONLY> bufr(context, &vr, 1);
      MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS> scratch(context, 2 * size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblasXnrm2<ScalarT>::func(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                                   scratch.dev(), 1, context._queue(), 0, NULL,
                                   &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(dot_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> buf1(context, size);
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> buf2(context, size);
      MemBuffer<ScalarT, CL_MEM_READ_ONLY> bufr(context, &vr, 1);
      MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS> scratch(context, size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblasXdot<ScalarT>::func(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                                  buf2.dev(), 0, 1, scratch.dev(), 1,
                                  context._queue(), 0, NULL, &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(iamax_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      unsigned vi;
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> buf1(context, size);
      MemBuffer<unsigned, CL_MEM_READ_ONLY> buf_i(context, &vi, 1);
      MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS> scratch(context, 2 * size);
      Event event;
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblasiXamax<ScalarT>::func(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                                    scratch.dev(), 1, context._queue(), 0, NULL,
                                    &event._cl());
        event.wait();
        event.release();
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(scal2op_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(context, size);
      MemBuffer<ScalarT> buf2(context, size);
      Event event1, event2;
      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblasXscal<ScalarT>::func(size, alpha, buf1.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event1._cl());
        clblasXscal<ScalarT>::func(size, alpha, buf2.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event2._cl());
        Event::wait({event1, event2});
        Event::release({event1, event2});
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(scal3op_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(context, size);
      MemBuffer<ScalarT> buf2(context, size);
      MemBuffer<ScalarT> buf3(context, size);
      Event event1, event2, event3;
      flops = benchmark<>::measure(no_reps, size * 3, [&]() {
        clblasXscal<ScalarT>::func(size, alpha, buf1.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event1._cl());
        clblasXscal<ScalarT>::func(size, alpha, buf2.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event2._cl());
        clblasXscal<ScalarT>::func(size, alpha, buf3.dev(), 0, 1, 1,
                                   context._queue(), 0, NULL, &event3._cl());
        Event::wait({event1, event2, event3});
        Event::release({event1, event2, event3});
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(axpy3op_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> bufsrc1(context, size);
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> bufsrc2(context, size);
      MemBuffer<ScalarT, CL_MEM_WRITE_ONLY> bufsrc3(context, size);
      MemBuffer<ScalarT> bufdst1(context, size);
      MemBuffer<ScalarT> bufdst2(context, size);
      MemBuffer<ScalarT> bufdst3(context, size);
      Event event1, event2, event3;
      flops = benchmark<>::measure(no_reps, size * 3, [&]() {
        clblasXaxpy<ScalarT>::func(size, alpha, bufsrc1.dev(), 0, 1,
                                   bufdst1.dev(), 0, 1, 1, context._queue(), 0,
                                   NULL, &event1._cl());
        clblasXaxpy<ScalarT>::func(size, alpha, bufsrc2.dev(), 0, 1,
                                   bufdst2.dev(), 0, 1, 1, context._queue(), 0,
                                   NULL, &event2._cl());
        clblasXaxpy<ScalarT>::func(size, alpha, bufsrc3.dev(), 0, 1,
                                   bufdst3.dev(), 0, 1, 1, context._queue(), 0,
                                   NULL, &event3._cl());
        Event::wait({event1, event2, event3});
        Event::release({event1, event2, event3});
      });
    }
    return flops;
  }

  BENCHMARK_FUNCTION(blas1_bench) {
    using ScalarT = TypeParam;
    double flops;
    {
      ScalarT alpha(3.135345123);
      ScalarT vr[4];
      unsigned vi;
      MemBuffer<ScalarT> buf1(context, size);
      MemBuffer<ScalarT> buf2(context, size);
      MemBuffer<ScalarT, CL_MEM_READ_ONLY> bufr(context, vr, 4);
      MemBuffer<unsigned, CL_MEM_READ_ONLY> buf_i(context, &vi, 1);

      MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS> scratch[4]{
          MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS>(context, size),
          MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS>(context, size),
          MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS>(context, 2 * size),
          MemBuffer<ScalarT, CL_MEM_HOST_NO_ACCESS>(context, 2 * size),
      };

      Event events[5];
      flops = benchmark<>::measure(no_reps, size * 12, [&]() {
        clblasXaxpy<ScalarT>::func(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0,
                                   1, 1, context._queue(), 0, NULL,
                                   &events[0]._cl());
        clblasXasum<ScalarT>::func(size, bufr.dev(), 0, buf2.dev(), 0, 1,
                                   scratch[0].dev(), 1, context._queue(), 0,
                                   NULL, &events[1]._cl());
        clblasXdot<ScalarT>::func(size, bufr.dev(), 1, buf1.dev(), 0, 1,
                                  buf2.dev(), 0, 1, scratch[1].dev(), 1,
                                  context._queue(), 0, NULL, &events[2]._cl());
        clblasXnrm2<ScalarT>::func(size, bufr.dev(), 2, buf1.dev(), 0, 1,
                                   scratch[2].dev(), 1, context._queue(), 0,
                                   NULL, &events[3]._cl());
        clblasiXamax<ScalarT>::func(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                                    scratch[3].dev(), 1, context._queue(), 0,
                                    NULL, &events[4]._cl());
        Event::wait({events[0], events[1], events[2], events[3], events[4]});
        Event::release({events[0], events[1], events[2], events[3], events[4]});
      });
    }
    return flops;
  }

  ~ClBlasBenchmarker() { clblasTeardown(); }
};

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 24, 10);
ClBlasBenchmarker blasbenchmark;

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
