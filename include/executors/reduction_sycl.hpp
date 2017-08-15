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
 *  @filename reduction_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS_REDUCTION_SYCL_HPP
#define BLAS_REDUCTION_SYCL_HPP

#include <evaluators/blas1_tree_evaluator.hpp>
#include <evaluators/blas2_tree_evaluator.hpp>
#include <evaluators/blas3_tree_evaluator.hpp>
#include <evaluators/blas_tree_evaluator.hpp>
#include <executors/blas1_tree_executor.hpp>
#include <executors/blas2_tree_executor.hpp>
#include <executors/blas3_tree_executor.hpp>
#include <executors/blas_device_sycl.hpp>
#include <executors/blas_packet_traits_sycl.hpp>

namespace blas {

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialised case for usingSharedMem == enabled, which contains a local
accessor.
@tparam ScalarT Value type of the local accessor.
*/
template <typename ScalarT>
struct shared_mem {
  using value_type = ScalarT;
  /*!
  @brief Template alias for a local accessor.
  @tparam valueT Value type of the accessor.
  */
  using local_accessor_t = cl::sycl::accessor<value_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

  /*!
  @brief Local accessor.
  */
  local_accessor_t localAcc;

  /*!
  @brief Constructor that creates a local accessor from a size and a SYCL
  command group handler.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  constexpr shared_mem(size_t size, cl::sycl::handler &h):
    localAcc(cl::sycl::range<1>(size), h)
  {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  value_type &operator[](cl::sycl::id<1> id) { return localAcc[id]; }
};

template <typename EvaluatorT, typename Scratch, typename ShMem, typename Functor>
struct KernelfES {
  using value_type = typename EvaluatorT::value_type;
  using dev_functor = Functor;

  EvaluatorT ev;
  Scratch scratch;
  ShMem shmem;

  KernelfES(EvaluatorT ev, Scratch scratch, ShMem shmem):
    ev(ev), scratch(scratch), shmem(shmem)
  {}

  void operator()(cl::sycl::nd_item<1> ndItem) {
    size_t i = ndItem.get_global(0);
    size_t N = ev.getSize();

    size_t localid = ndItem.get_local(0);
    size_t groupid = ndItem.get_group(0);
    size_t localsize = ndItem.get_local_range(0);
    size_t nwg = ndItem.get_num_groups(0);

    // Reduction across the grid
    value_type val = dev_functor::template init<value_type>();
    for (size_t k = 2 * groupid * localsize + localid; k < N; k += 2 * localsize * nwg) {
      val = dev_functor::eval(val, ev.eval(k));
      if (k + localsize < N) {
        val = dev_functor::eval(val, ev.eval(k + localsize));
      }
    }

    shmem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localsize >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        value_type l=shmem[localid],r=shmem[localid+offset];
        shmem[localid] = dev_functor::eval(shmem[localid], shmem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      scratch[groupid] = shmem[localid];
    }
  }
};

template <typename Result, typename Scratch, typename ShMem, typename EvaluatorT, typename Functor>
struct KernelfSR {
  using value_type = typename EvaluatorT::value_type;
  using dev_functor = Functor;

  Result result;
  Scratch scratch;
  ShMem shmem;
  size_t size;

  KernelfSR(Result result, Scratch scratch, ShMem shmem, size_t size):
    result(result), scratch(scratch), shmem(shmem), size(size)
  {}

  void operator()(cl::sycl::nd_item<1> ndItem) {
    size_t i = ndItem.get_global(0);
    size_t localid = ndItem.get_local(0);
    size_t groupid = ndItem.get_group(0);
    size_t localsize = ndItem.get_local_range(0);
    size_t nwg = ndItem.get_num_groups(0);

    // Reduction across the grid
    shmem[localid] = scratch[localid];

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localsize >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        shmem[localid] = dev_functor::eval(shmem[localid], shmem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      result[groupid] = shmem[localid];
    }
  }
};

template <class EvaluatorT, class Functor>
struct GenericReducerTwoStage : GenericReducer<EvaluatorT, Functor> {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void execute1(Device &dev, cl::sycl::buffer<value_type, 1> scratch, EvaluatorT ev, size_t localsize, size_t globalsize, size_t sharedsize) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      auto acc = scratch.template get_access<cl::sycl::access::mode::write>(h);
      shared_mem<value_type> shmem(sharedsize, h);
      KernelfES<decltype(nTree), decltype(acc), decltype(shmem), Functor> kf(nTree, acc, shmem);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration, kf);
    });
    dev.sycl_queue().wait_and_throw();
  }

  static void execute2(Device &dev, cl::sycl::buffer<value_type, 1> result, cl::sycl::buffer<value_type, 1> scratch, size_t localsize, size_t globalsize, size_t sharedsize, size_t N) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto acc_scr = scratch.template get_access<cl::sycl::access::mode::read>(h);
      auto acc_res = result.template get_access<cl::sycl::access::mode::write>(h);
      shared_mem<value_type> shmem(sharedsize, h);
      KernelfSR<decltype(acc_res), decltype(acc_scr), decltype(shmem), EvaluatorT, Functor> kf(acc_res, acc_scr, shmem, N);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration, kf);
    });
    dev.sycl_queue().wait_and_throw();
  }

  static void run(Device &dev, EvaluatorT ev, cl::sycl::buffer<value_type, 1> result) {
    size_t N = ev.getSize();
    size_t localsize = 256;
    size_t nwg = 512;
    size_t sharedsize = nwg;
    size_t globalsize = localsize * nwg;
    cl::sycl::buffer<value_type, 1> scratch{nwg};
    execute1(dev, scratch, ev, localsize, globalsize, sharedsize);
    execute2(dev, result, scratch, nwg, nwg, sharedsize, (N + localsize - 1) / localsize);
  }
};

template <class EvaluatorT, typename Functor>
struct GenericReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT ev, cl::sycl::buffer<value_type, 1> result) {
    GenericReducerTwoStage<EvaluatorT, Functor>::run(dev, ev, result);
  }
};

template <class EvaluatorT, class Functor, class Reducer>
struct FullReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {}
};

template <class EvaluatorT, class Functor, class Reducer>
struct PartialReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {}
};

}  // namespace BLAS

#endif
