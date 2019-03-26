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
 *  @filename clblast_benchmark.cpp
 *
 **************************************************************************/

#include <complex>
#include <vector>

#include <clblast.h>

#include "../common/blas_benchmark.hpp"

BENCHMARK_NAME_FORMAT(clblast_level_1) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_" << params;
  return fname.str();
}

BENCHMARK(scal, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  // Alpha
  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  std::vector<scalar_t> buf1_host = benchmark<>::random_data<scalar_t>(size);
  MemBuffer<scalar_t> buf1(*ex, buf1_host.data(), size);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 1, [&]() {
    clblast::Scal<scalar_t>(size, alpha, buf1.dev(), 0, 1, (*ex)._queue(),
                           &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(axpy, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  std::vector<scalar_t> buf1_host = benchmark<>::random_data<scalar_t>(size);
  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(*ex, buf1_host.data(), size);

  std::vector<scalar_t> buf2_host = benchmark<>::random_data<scalar_t>(size);
  MemBuffer<scalar_t> buf2(*ex, buf2_host.data(), size);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Axpy<scalar_t>(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0, 1,
                           (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(asum, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t vr;
  std::vector<scalar_t> buf1_host = benchmark<>::random_data<scalar_t>(size);
  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(*ex, buf1_host.data(), size);
  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufr(*ex, &vr, 1);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Asum<scalar_t>(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                           (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(nrm2, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t vr;

  std::vector<scalar_t> buf1_host = benchmark<>::random_data<scalar_t>(size);
  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(*ex, buf1_host.data(), size);
  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufr(*ex, &vr, 1);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Nrm2<scalar_t>(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                           (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(dot, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t vr;
  std::vector<scalar_t> buf1_host = benchmark<>::const_data<scalar_t>(size);
  std::vector<scalar_t> buf2_host = benchmark<>::const_data<scalar_t>(size);

  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(*ex, buf1_host.data(), size);
  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf2(*ex, buf2_host.data(), size);
  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufr(*ex, &vr, 1);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Dot<scalar_t>(size, bufr.dev(), 0, buf1.dev(), 0, 1, buf2.dev(), 0,
                          1, (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(iamax, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  int vi;
  auto buf1_host = benchmark<>::const_data<scalar_t>(size);

  MemBuffer<scalar_t, CL_MEM_WRITE_ONLY> buf1(*ex, buf1_host.data(), size);
  MemBuffer<int, CL_MEM_READ_ONLY> buf_i(*ex, &vi, 1);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Amax<scalar_t>(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                           (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

// not supported at current release yet
BENCHMARK(iamin, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  int vi;
  auto buf1_host = benchmark<>::random_data<scalar_t>(size);

  MemBuffer<scalar_t> buf1(*ex, buf1_host.data(), size);
  MemBuffer<int> buf_i(*ex, &vi, 1);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Amin<scalar_t>(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                           (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

BENCHMARK(scal2op, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();
  auto buf1_host = benchmark<>::random_data<scalar_t>(size);
  auto buf2_host = benchmark<>::random_data<scalar_t>(size);

  MemBuffer<scalar_t> buf1(*ex, buf1_host.data(), size);
  MemBuffer<scalar_t> buf2(*ex, buf2_host.data(), size);

  Event event1, event2;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 2, [&]() {
    clblast::Scal<scalar_t>(size, alpha, buf1.dev(), 0, 1, (*ex)._queue(),
                           &event1._cl());
    clblast::Scal<scalar_t>(size, alpha, buf2.dev(), 0, 1, (*ex)._queue(),
                           &event2._cl());
    Event::wait({event1, event2});
    Event::release({event1});
    return event2;
  });

  return flops;
}

BENCHMARK(scal3op, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();

  auto buf1_host = benchmark<>::random_data<scalar_t>(size);
  auto buf2_host = benchmark<>::random_data<scalar_t>(size);
  auto buf3_host = benchmark<>::random_data<scalar_t>(size);

  MemBuffer<scalar_t> buf1(*ex, buf1_host.data(), size);
  MemBuffer<scalar_t> buf2(*ex, buf2_host.data(), size);
  MemBuffer<scalar_t> buf3(*ex, buf3_host.data(), size);

  Event event1, event2, event3;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, size * 3, [&]() {
    clblast::Scal<scalar_t>(size, alpha, buf1.dev(), 0, 1, (*ex)._queue(),
                           &event1._cl());
    clblast::Scal<scalar_t>(size, alpha, buf2.dev(), 0, 1, (*ex)._queue(),
                           &event2._cl());
    clblast::Scal<scalar_t>(size, alpha, buf3.dev(), 0, 1, (*ex)._queue(),
                           &event3._cl());
    Event::wait({event1, event2, event3});
    Event::release({event1, event2});
    return event3;
  });

  return flops;
}

BENCHMARK(axpy3op, clblast_level_1) {
  using scalar_t = ElemT;
  const size_t size = params;

  scalar_t alphas[] = {1.78426458744, 2.187346575843, 3.78164387328};
  size_t offsets[] = {0, size, size * 2};

  auto bufscr_host = benchmark<>::random_data<scalar_t>(size * 3);
  auto bufdst_host = benchmark<>::random_data<scalar_t>(size * 3);

  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufsrc(*ex, bufscr_host.data(),
                                              size * 3);
  MemBuffer<scalar_t> bufdst(*ex, bufdst_host.data(), size * 3);

  Event event;
  benchmark<>::datapoint_t flops =
      benchmark<>::measure(reps, size * 3 * 2, [&]() {
        clblast::AxpyBatched<scalar_t>(size, alphas, bufsrc.dev(), offsets, 1,
                                      bufdst.dev(), offsets, 1, 3,
                                      (*ex)._queue(), &event._cl());
        event.wait();
        return event;
      });

  return flops;
}

SUITE(ADD(scal), ADD(axpy), ADD(nrm2), ADD(dot), ADD(iamax), ADD(iamin),
      ADD(scal2op), ADD(scal3op), ADD(axpy3op))

CLBLAST_BENCHMARK_MAIN(default_ranges::level_1, 10)