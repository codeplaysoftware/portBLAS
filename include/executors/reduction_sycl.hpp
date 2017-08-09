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
struct scratch_mem {
  using value_type = ScalarT;
  /*!
  @brief Template alias for a local accessor.
  @tparam valueT Value type of the accessor.
  */
  using local_accessor_t =
      cl::sycl::accessor<value_type, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;

  /*!
  @brief Local accessor.
  */
  const size_t size;
  local_accessor_t localAcc;

  /*!
  @brief Constructor that creates a local accessor from a size and a SYCL
  command group handler.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  constexpr scratch_mem(size_t size, cl::sycl::handler &h)
      : localAcc(cl::sycl::range<1>(size), h), size(size) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  value_type &operator[](cl::sycl::id<1> id) { return localAcc[id]; }
};

template <typename EvaluatorT, typename Scratch>
struct KernelFunctor {
  EvaluatorT ev;
  Scratch scratch;

  KernelFunctor(EvaluatorT ev, Scratch scratch) : ev(ev), scratch(scratch) {}

  void operator()(cl::sycl::nd_item<1> ndItem) {
    using dev_functor = typename EvaluatorT::dev_functor;
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);

    /*       size_t vecS = ev.r.getSize(); */
    /*       size_t frs_thrd = 2 * groupid * localSz + localid; */
    /*       size_t blqS = 256; */
    /*       size_t grdS = 512; */

    /*       // Reduction across the grid */
    /*       value_type val = dev_functor::init(ev.r); */
    /*       for (size_t k = frs_thrd; k < vecS; k += 2 * grdS) { */
    /*         val = dev_functor::eval(val, ev.r.eval(k)); */
    /*         if ((k + blqS < vecS)) { */
    /*           val = dev_functor::eval(val, ev.r.eval(k + blqS)); */
    /*         } */
    /*       } */

    /*       scratch[localid] = val; */
    /*       // This barrier is mandatory to be sure the data is on the shared
     * memory */
    /*       ndItem.barrier(cl::sycl::access::fence_space::local_space); */

    /*       // Reduction inside the block */
    /*       for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) { */
    /*         if (localid < offset) { */
    /*           scratch[localid] = dev_functor::eval(scratch[localid],
     * scratch[localid + offset]); */
    /*         } */
    /*         // This barrier is mandatory to be sure the data are on the
     * shared */
    /*         // memory */
    /*         ndItem.barrier(cl::sycl::access::fence_space::local_space); */
    /*       } */
    /*       if (localid == 0) { */
    /*         ev.l.eval(groupid) = scratch[localid]; */
    /*       } */
    /* l.evalref(i) = r.subeval(groupid); // similar to the line below? */
    /* return ev.l.eval(groupid); */
  }
};

template <class EvaluatorT>
struct GenericReducerTwoStage : GenericReducer<EvaluatorT> {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT ev,
                  cl::sycl::buffer<value_type, 1> result) {
    /* Evaluator<ExpressionT, Device> ev(expr); */
    /* size_t localsize = 256; */
    /* size_t nwg = 512; */
    /* size_t sharedsize = nwg; */
    /* size_t globalsize = ev.getSize(); */
    /* auto sh = make_scratchMem(localsize); */
    /* execute<KernelFunctor>(dev, ev, sh, localsize, localsize * nwg,
     * sharedsize); */
    /* execute<KernelFunctor>(dev, sh, ev, localsize, localsize); */
  }
};

template <class EvaluatorT>
struct GenericReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT ev,
                  cl::sycl::buffer<value_type, 1> result) {
    GenericReducerTwoStage<EvaluatorT>::run(dev, ev, result);
  }
};

template <class EvaluatorT, class Reducer>
struct FullReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev,
                  cl::sycl::buffer<value_type, 1> result) {
    /* Reducer::reduce(ev, dev, result); */
  }
};

template <class EvaluatorT, class Reducer>
struct PartialReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev,
                  cl::sycl::buffer<value_type, 1> result) {}
};

}  // namespace BLAS

#endif
