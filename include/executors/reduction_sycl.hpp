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
#include <executors/reduction_acc_traits.hpp>

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
  @param h SYCL command group handler.
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

template <typename AssignEvaluatorT, typename EvaluatorT, typename Result, typename ShMem, typename Functor>
struct KernelfOneStage {
  using value_type = typename EvaluatorT::value_type;
  using dev_functor = Functor;

  AssignEvaluatorT assign;
  EvaluatorT ev;
  Result result;
  ShMem shmem;

  KernelfOneStage(AssignEvaluatorT assign, EvaluatorT ev, Result result, ShMem shmem):
    assign(assign), ev(ev), result(result), shmem(shmem)
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
    #pragma unroll
    for (size_t k = i; k < N; k += localsize * nwg) {
      accum_functor_traits<dev_functor>::acc(val, ev.eval(k));
    }

    shmem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    /* // Reduction inside the block */
    #pragma unroll
    for (size_t offset = localsize >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        value_type l=shmem[localid],r=shmem[localid+offset];
        shmem[localid] = dev_functor::eval(shmem[localid], shmem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      result[groupid] = shmem[localid];
      assign.template eval_<!std::is_same<
        typename std::remove_reference<decltype(assign.r.r)>::type,
        typename std::remove_reference<decltype(ev)>::type
      >::value>(0);
    }
  }
};

template <typename EvaluatorT, typename Scratch, typename ShMem, typename Functor>
struct KernelfFirstStage {
  using value_type = typename EvaluatorT::value_type;
  using dev_functor = Functor;

  EvaluatorT ev;
  Scratch scratch;
  ShMem shmem;

  KernelfFirstStage(EvaluatorT ev, Scratch scratch, ShMem shmem):
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
    #pragma unroll
    for (size_t k = i; k < N; k += localsize * nwg) {
      accum_functor_traits<dev_functor>::acc(val, ev.eval(k));
    }

    shmem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    #pragma unroll
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

template <typename AssignEvaluatorT, typename Result, typename Scratch, typename ShMem, typename EvaluatorT, typename Functor>
struct KernelfSecondStage {
  using value_type = typename EvaluatorT::value_type;
  using dev_functor = Functor;

  AssignEvaluatorT assign;
  Result result;
  Scratch scratch;
  ShMem shmem;
  size_t size;

  KernelfSecondStage(AssignEvaluatorT assign, Result result, Scratch scratch, ShMem shmem, size_t size):
    assign(assign), result(result), scratch(scratch), shmem(shmem), size(size)
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
    #pragma unroll
    for (size_t offset = localsize >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        shmem[localid] = dev_functor::eval(shmem[localid], shmem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      result[groupid] = shmem[localid];
      assign.template eval_<false>(0);
    }
  }
};

#ifndef RETURNCXX11
#define RETURNCXX11(expr) -> decltype(expr) { return expr; }
#endif

namespace detail {
  class NoEvaluation {
    using Device = SYCLDevice;
    template <bool NEED_ASSIGN>
    void eval_(size_t i) {}
    template <bool NEED_ASSIGN>
    void eval_(cl::sycl::nd_item<1> ndItem) {}
  };

  template <typename AssignEvaluatorT>
  struct assign_trait {
    using type = AssignEvaluatorT;
  };
  template <>
  struct assign_trait<void> {
    using type = NoEvaluation;
  };

  template <typename AssignEvaluatorT>
  struct DereferenceAssign {
    static auto deref(AssignEvaluatorT *assign, cl::sycl::handler &h) RETURNCXX11(
      blas::make_accessor(*assign, h)
    )
  };

  template <>
  struct DereferenceAssign<void> {
    static typename detail::assign_trait<void>::type deref(void *assign, cl::sycl::handler &h) {
      return NoEvaluation{};
    }
  };
} // namespace detail

template <class AssignEvaluatorT>
auto dereference(AssignEvaluatorT *assign, cl::sycl::handler &h) RETURNCXX11(
  detail::DereferenceAssign<AssignEvaluatorT>::deref(assign, h)
)

template <class AssignEvaluatorT, class EvaluatorT, class Functor>
struct GenericReducerOneStage {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void execute(Device &dev, AssignEvaluatorT *assign, cl::sycl::buffer<value_type, 1> result, const EvaluatorT &ev, size_t localsize, size_t globalsize, size_t sharedsize) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      auto asgn = dereference(assign, h);
      auto acc = result.template get_access<cl::sycl::access::mode::write>(h);
      shared_mem<value_type> shmem(sharedsize, h);
      KernelfOneStage<decltype(asgn), decltype(nTree), decltype(acc), decltype(shmem), Functor> kf(asgn, nTree, acc, shmem);
      h.parallel_for(
        cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize},
          cl::sycl::range<1>{localsize}
        }, kf
      );
    });
  }

  static void run(Device &dev, AssignEvaluatorT *assign, const EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {
    size_t N = ev.getSize();
    size_t localsize;
    size_t nwg_noset;
    size_t globalsize;
    dev.template generic_reduction_setup<value_type>(localsize, nwg_noset, globalsize, N);
    size_t nwg = 1;
    size_t sharedsize = localsize;
    globalsize = localsize * nwg;
    execute(dev, assign, result, ev, localsize, globalsize, sharedsize);
  }
};

template <class AssignEvaluatorT, class EvaluatorT, class Functor>
struct GenericReducerTwoStages {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void execute1(Device &dev, cl::sycl::buffer<value_type, 1> scratch, const EvaluatorT &ev, size_t localsize, size_t globalsize, size_t sharedsize) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      auto acc = scratch.template get_access<cl::sycl::access::mode::write>(h);
      shared_mem<value_type> shmem(sharedsize, h);
      KernelfFirstStage<decltype(nTree), decltype(acc), decltype(shmem), Functor> kf(nTree, acc, shmem);
      h.parallel_for(
        cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize},
          cl::sycl::range<1>{localsize}
        }, kf
      );
    });
    dev.sycl_queue().wait_and_throw();
  }

  static void execute2(Device &dev, AssignEvaluatorT *assign, cl::sycl::buffer<value_type, 1> result, cl::sycl::buffer<value_type, 1> scratch, size_t localsize, size_t globalsize, size_t sharedsize, size_t N) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto asgn = dereference(assign, h);
      auto acc_scr = scratch.template get_access<cl::sycl::access::mode::read>(h);
      auto acc_res = result.template get_access<cl::sycl::access::mode::write>(h);
      shared_mem<value_type> shmem(sharedsize, h);
      KernelfSecondStage<decltype(asgn), decltype(acc_res), decltype(acc_scr), decltype(shmem), EvaluatorT, Functor> kf(asgn, acc_res, acc_scr, shmem, N);
      h.parallel_for(
        cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize},
          cl::sycl::range<1>{localsize}
        }, kf
      );
    });
    dev.sycl_queue().wait_and_throw();
  }

  static void run(Device &dev, AssignEvaluatorT *assign, const EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {
    size_t N = ev.getSize();
    size_t localsize;
    size_t nwg;
    size_t globalsize;
    dev.template generic_reduction_setup<value_type>(localsize, nwg, globalsize, N);
    size_t sharedsize = localsize;
    cl::sycl::buffer<value_type, 1> scratch{nwg};
    execute1(dev, scratch, ev, localsize, globalsize, sharedsize);
    execute2(dev, assign, result, scratch, nwg, nwg, sharedsize=nwg, (N + localsize - 1) / localsize);
  }
};

/*!
 * @brief Generic reducer between dimensions (currently just vector -> scalar).
 */
template <class AssignEvaluatorT, class EvaluatorT, class Functor>
struct GenericReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, AssignEvaluatorT *assign, const EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {
    if(dev.sycl_device().is_gpu()) {
      GenericReducerTwoStages<AssignEvaluatorT, EvaluatorT, Functor>::run(dev, assign, ev, result);
    } else if(dev.sycl_device().is_cpu() || dev.sycl_device().is_host()) {
      GenericReducerOneStage<AssignEvaluatorT, EvaluatorT, Functor>::run(dev, assign, ev, result);
    } else {
      throw std::runtime_error("unsupported device");
    }
  }
};

template <class EvaluatorT, typename Functor>
struct GenericReducer<void, EvaluatorT, Functor> {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, void *assign, const EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {
    if(dev.sycl_device().is_gpu() || ev.getSize() > (1<<24)) {
      GenericReducerTwoStages<void, EvaluatorT, Functor>::run(dev, assign, ev, result);
    } else if(dev.sycl_device().is_cpu() || dev.sycl_device().is_host()) {
      GenericReducerOneStage<void, EvaluatorT, Functor>::run(dev, assign, ev, result);
    } else {
      throw std::runtime_error("unsupported device");
    }
  }
};

/*!
 * @brief Not implemented (vector/matrix -> scalar).
 */
template <class EvaluatorT, class Functor, class Reducer>
struct FullReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {}
};

/*!
 * @brief Not implemented (matrix -> vector).
 */
template <class EvaluatorT, class Functor, class Reducer>
struct PartialReducer {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;

  static void run(Device &dev, EvaluatorT &ev, cl::sycl::buffer<value_type, 1> result) {}
};

}  // namespace BLAS

#endif
