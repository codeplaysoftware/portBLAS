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
 *  @filename blas3_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS3_INTERFACE_SYCL_GEMM_HPP
#define BLAS3_INTERFACE_SYCL_GEMM_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cctype>

// #define VERBOSE 1

#include <executors/executor_sycl.hpp>
#include <operations/blas3_trees_gemm.hpp>

using namespace cl::sycl;

namespace blas {

template <typename T> struct Wrap {};

template <typename Gemm, typename ExecutorType, typename T>
inline void _gemm_v2_tr(
  Executor<ExecutorType> ex,
  int _M, int _N, int _K, T _alpha,
  cl::sycl::buffer<T, 1> _A, int _lda,
  cl::sycl::buffer<T, 1> _B, int _ldb,
  T _beta,
  cl::sycl::buffer<T, 1> _C, int _ldc)
{
  ex.sycl_queue().submit([&](handler &h) {
    auto accA = _A.template get_access<access::mode::read>(h);
    auto accB = _B.template get_access<access::mode::read>(h);
    auto accC = _C.template get_access<access::mode::read_write>(h);
    h.parallel_for<Wrap<Gemm>>(Gemm::get_nd_range(_M, _N),
        [=](nd_item<1> id) {
          Gemm::run(id.get_global(0), _M, _N, _K, T(_alpha),
                    accA.get_pointer(), _lda, accB.get_pointer(), _ldb,
                    T(_beta), accC.get_pointer(), _ldc);
    });
  });
  ex.sycl_queue().wait();
}

template <size_t WG, typename ExecutorType, typename T>
inline void _gemm_v2(
  Executor<ExecutorType> ex,
  bool _TransA, bool _TransB,
  int _M, int _N, int _K, T _alpha,
  cl::sycl::buffer<T, 1> _A, int _lda,
  cl::sycl::buffer<T, 1> _B, int _ldb,
  T _beta,
  cl::sycl::buffer<T, 1> _C, int _ldc)
{
  if(_TransA && _TransB) {
    _gemm_v2_tr<GemmFactoryV2<WG, true, true, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _beta, _C, _ldc);
    return;
  } else if(_TransA && !_TransB) {
    _gemm_v2_tr<GemmFactoryV2<WG, true, false, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _beta, _C, _ldc);
    return;
  } else if(!_TransA && _TransB) {
    _gemm_v2_tr<GemmFactoryV2<WG, false, true, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _beta, _C, _ldc);
    return;
  } else if(!_TransA && !_TransB) {
    _gemm_v2_tr<GemmFactoryV2<WG, false, false, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _beta, _C, _ldc);
    return;
  }
  throw std::runtime_error("leak in the condition");
}

template <typename Gemm, typename ExecutorType, typename T>
void _gemm_v19_tr(
  Executor<ExecutorType> ex,
  int _M, int _N, int _K, T _alpha,
  cl::sycl::buffer<T, 1> _A, int _lda,
  cl::sycl::buffer<T, 1> _B, int _ldb,
  T _beta,
  cl::sycl::buffer<T, 1> _C, int _ldc)
{
  ex.sycl_queue().submit([&] (handler &h) {
    auto accA = _A.template get_access<access::mode::read>(h);
    auto accB = _B.template get_access<access::mode::read>(h);
    auto accC = _C.template get_access<access::mode::read_write>(h);
    accessor<T, 1, access::mode::read_write, access::target::local> scratch(range<1>(Gemm::scratch_size), h);
    h.parallel_for<Wrap<Gemm>>(Gemm::get_nd_range(_M, _N),
        [=](nd_item<1> id) {
          Gemm::run(
            id, id.get_group(0), id.get_local(0), _M, _N, _K, T(_alpha),
            accA.get_pointer(), _lda, accB.get_pointer(), _ldb, T(_beta),
            accC.get_pointer(), _ldc, scratch.get_pointer());
    });
  });
  ex.sycl_queue().wait();
}

template <bool DoubleBuffer, size_t ClSize, typename TileT, typename ExecutorType, typename T>
void _gemm_v19(
  Executor<ExecutorType> ex,
  bool _TransA, bool _TransB,
  int _M, int _N, int _K, T _alpha,
  cl::sycl::buffer<T, 1> _A, int _lda,
  cl::sycl::buffer<T, 1> _B, int _ldb,
  T _beta,
  cl::sycl::buffer<T, 1> _C, int _ldc)
{
  if(_TransA && _TransB) {
    _gemm_v19_tr<GemmFactoryV19<DoubleBuffer, ClSize, TileT, true, true, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    return;
  } else if(_TransA && !_TransB) {
    _gemm_v19_tr<GemmFactoryV19<DoubleBuffer, ClSize, TileT, true, false, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    return;
  } else if(!_TransA && _TransB) {
    _gemm_v19_tr<GemmFactoryV19<DoubleBuffer, ClSize, TileT, false, true, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    return;
  } else if(!_TransA && !_TransB) {
    _gemm_v19_tr<GemmFactoryV19<DoubleBuffer, ClSize, TileT, false, false, T>>(ex, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    return;
  }
  throw std::runtime_error("leak in the condition");
}

typedef enum { UNSUPPORTED_DEVICE, INTELGPU, AMDGPU } DEVICETYPE;
DEVICETYPE get_device_type(const cl::sycl::device dev) {
  //std::vector<cl::sycl::device> supported_devices;
  cl::sycl::platform platform = dev.get_platform();
  auto plat_name = platform.template get_info<cl::sycl::info::platform::name>();
  std::transform(plat_name.begin(), plat_name.end(), plat_name.begin(), ::tolower);
  if(plat_name.find("amd")!=std::string::npos && dev.is_gpu()) {
  	return AMDGPU;
  } else if(plat_name.find("intel")!=std::string::npos && dev.is_gpu()) {
  	return INTELGPU;
  } else {
  	return INTELGPU;//UNSUPPORTED_DEVICE;
  }
  throw std::runtime_error("couldn't find device");
}

template <typename ExecutorType, typename T>
void _gemm(Executor<ExecutorType> ex, char _TransA, char _TransB,
           int _M, int _N, int _K, T _alpha,
           cl::sycl::buffer<T, 1> _A, int _lda,
           cl::sycl::buffer<T, 1> _B, int _ldb,
           T _beta,
           cl::sycl::buffer<T, 1> _C, int _ldc)
{
  #define ARGS ex, _TransA, _TransB, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc
  #define IF_MNK_EQ(m,n,k) if(_M==(m) && _N==(n) && _K==(k))
  cl::sycl::device dev = ex.sycl_queue().get_device();
  if(get_device_type(dev) == AMDGPU) {
    IF_MNK_EQ(4096, 4096, 4096) {
      _gemm_v19<false, 64, Tile<8,8,16,16>>(ARGS);
      return;
    } else IF_MNK_EQ(2048, 784, 1024) {
      _gemm_v19<false, 64, Tile<8,8,16,16>>(ARGS);
      return;
    } else IF_MNK_EQ(4096, 2048, 1024) {
      _gemm_v19<false, 64, Tile<8,8,16,16>>(ARGS);
      return;
    } else IF_MNK_EQ(1024, 4096, 1024) {
      _gemm_v19<false, 64, Tile<8,8,16,16>>(ARGS);
      return;
    } else IF_MNK_EQ(10, 1024, 1024) {
      _gemm_v19<true, 64, Tile<1,1,16,16>>(ARGS);
      return;
    }
  } else if(get_device_type(dev) == INTELGPU) {
    IF_MNK_EQ(4096, 4096, 4096) {
      _gemm_v19<false, 64, Tile<8,8,8,8>>(ARGS);
      return;
    } else IF_MNK_EQ(2048, 784, 1024) {
      _gemm_v19<false, 64, Tile<8,8,8,8>>(ARGS);
      return;
    } else IF_MNK_EQ(4096, 2048, 1024) {
      _gemm_v19<false, 64, Tile<8,8,8,8>>(ARGS);
      return;
    } else IF_MNK_EQ(1024, 4096, 1024) {
      _gemm_v19<false, 64, Tile<4,4,16,16>>(ARGS);
      return;
    } else IF_MNK_EQ(10, 1024, 1024) {
      _gemm_v19<false, 64, Tile<2,2,8,8>>(ARGS);
      return;
    }
  }
  throw std::runtime_error("not implemented");
}

}  // namespace blas

#endif  // BLAS3_INTERFACE_SYCL_HPP
