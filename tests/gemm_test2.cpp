/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  @filename gemm_test2.hpp
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <vector>

#include <CL/sycl.hpp>


#include <operations/blas3_trees_gemm.hpp>


#include "test_utils.hpp"


using namespace cl::sycl;
using namespace blas;


#define ENABLE_SYSTEM_GEMM(_type, _system_name) \
extern "C" void _system_name( \
    const char *, const char *, const int *, const int *, const int *, \
    const _type *, const _type *, const int *, const _type *, const int *, \
    const _type *, _type *, const int *); \
void gemm( \
    const char *transA, const char *transB, int m, int n, int k, _type alpha, \
    const _type a[], int lda, const _type b[], int ldb, _type beta, \
    _type c[], int ldc) { \
  _system_name(transA, transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, \
               c, &ldc); \
}

ENABLE_SYSTEM_GEMM(float, sgemm_)
ENABLE_SYSTEM_GEMM(double, dgemm_)

#undef ENABLE_SYSTEM_GEMM


template <typename T> class Test {};


template <typename Gemm, typename Container>
typename std::enable_if<Gemm::version == 2>::type
test(int r, int m, int n, int k, const Container &dataA,
     const Container &dataB, Container dataC, const Container &refC,
     cl::sycl::queue q)
{
  using etype = typename Gemm::value_type;
  std::cout << "\n=== Testing " << Gemm::get_type_string() << " ==="
            << std::endl;
  {
    buffer<etype, 1> buffA(dataA.data(), range<1>(dataA.size()));
    buffer<etype, 1> buffB(dataB.data(), range<1>(dataB.size()));
    buffer<etype, 1> buffC(dataC.data(), range<1>(dataC.size()));
    run_test(r, 2.0*m*n*k, [&] {
      q.submit([&] (handler &cgh) {
        auto accA = buffA.template get_access<access::mode::read>(cgh);
        auto accB = buffB.template get_access<access::mode::read>(cgh);
        auto accC = buffC.template get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<Test<Gemm>>(Gemm::get_nd_range(m, n),
            [=](nd_item<1> id) {
          Gemm::run(id.get_global(0), m, n, k, etype(1.0),
                    accA.get_pointer(), m, accB.get_pointer(), k, etype(1.0),
                    accC.get_pointer(), m);
        });
      });
      q.wait();
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}


template <typename Gemm, typename Container>
typename std::enable_if<Gemm::version == 19>::type
test(int r, int m, int n, int k, const Container &dataA,
     const Container &dataB, Container dataC, const Container &refC,
     cl::sycl::queue q)
{
  using etype = typename Gemm::value_type;
  std::cout << "\n=== Testing " << Gemm::get_type_string() << " ==="
            << std::endl;
  {
    buffer<etype, 1> buffA(dataA.data(), range<1>(dataA.size()));
    buffer<etype, 1> buffB(dataB.data(), range<1>(dataB.size()));
    buffer<etype, 1> buffC(dataC.data(), range<1>(dataC.size()));
    run_test(r, 2.0*m*n*k, [&] {
      q.submit([&] (handler &cgh) {
        auto accA = buffA.template get_access<access::mode::read>(cgh);
        auto accB = buffB.template get_access<access::mode::read>(cgh);
        auto accC = buffC.template get_access<access::mode::read_write>(cgh);
        accessor<etype, 1, access::mode::read_write, access::target::local>
          scratch(range<1>(Gemm::scratch_size), cgh);
        cgh.parallel_for<Test<Gemm>>(Gemm::get_nd_range(m, n),
            [=](nd_item<1> id) {
          Gemm::run(id, id.get_group(0), id.get_local(0), m, n, k, etype(1.0),
                    accA.get_pointer(), m, accB.get_pointer(), k, etype(1.0),
                    accC.get_pointer(), m, scratch.get_pointer());
        });
      });
      q.wait();
    });
  }
  std::cout << "err  = " << relative_diff(refC, dataC) << std::endl;
}


int main(int argc, char *argv[]) {
  using etype = float;
  const int seed = 42;

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " M N K rep" << std::endl;
    return -1;
  }

  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);
  const int rep = std::atoi(argv[4]);

  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<etype>(m, k, -1, 1, rnd);
  auto dataB = gen_matrix<etype>(k, n, -1, 1, rnd);
  auto origC = gen_matrix<etype>(m, n, -1, 1, rnd);
  auto refC = origC;


  std::cout << "\n=== Testing system CPU implementation ===" << std::endl;
  run_test(rep, 2.0*m*n*k, [&] {
    gemm("N", "N", m, n, k, etype(1), dataA.data(), m, dataB.data(), k,
         etype(1), refC.data(), m);
  });

  cl::sycl::queue q;
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;


#define DATA rep, m, n, k, dataA, dataB, origC, refC, q

  const int cls = 64; // size of cache line in bytes
  const bool db = true; // use double buffer
  const bool ta = true; // transpose A
  const bool tb = true; // transpose B

  test<GemmFactoryV2<128, !ta, !tb, etype>>(DATA);


  test<GemmFactoryV19<!db, cls, Tile<1, 1, 8, 8>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<1, 1, 8, 8>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<2, 2, 8, 8>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<2, 2, 8, 8>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<4, 4, 8, 8>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<4, 4, 8, 8>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<8, 8, 8, 8>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<8, 8, 8, 8>, !ta, !tb, etype>>(DATA);


  test<GemmFactoryV19<!db, cls, Tile<1, 1, 16, 16>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<1, 1, 16, 16>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<2, 2, 16, 16>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<2, 2, 16, 16>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<4, 4, 16, 16>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<4, 4, 16, 16>, !ta, !tb, etype>>(DATA);

  test<GemmFactoryV19<!db, cls, Tile<8, 8, 16, 16>, !ta, !tb, etype>>(DATA);
  test<GemmFactoryV19<db, cls, Tile<8, 8, 16, 16>, !ta, !tb, etype>>(DATA);

#undef DATA

  return 0;
}

