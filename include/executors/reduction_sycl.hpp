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
  /*!
  @brief Template alias for a local accessor.
  @tparam valueT Value type of the accessor.
  */
  using local_accessor_t =
      cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;

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
  shared_mem(size_t size, cl::sycl::handler &cgh)
      : localAcc(cl::sycl::range<1>(size), cgh) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  ScalarT &operator[](cl::sycl::id<1> id) { return localAcc[id]; }
};

template <typename EvaluatorT>
struct KernelFunctorNoScratch {
  EvaluatorT ev;

  KernelFunctorNoScratch(EvaluatorT ev) : ev(ev) {}

  void operator()(cl::sycl::nd_item<1> ndItem) {
    using dev_functor = typename EvaluatorT::dev_functor;
    /* size_t vecS = ev.r.getSize(); */

    /* size_t frs_thrd = 2 * blqS * i; */
    /* size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd +
     * blqS); */
    /* // Reduction across the grid */
    /* value_type val = dev_functor::init(ev.r); */
    /* for (size_t j = frs_thrd; j < lst_thrd; j++) { */
    /*   value_type local_val = dev_functor::init(ev.r); */
    /*   for (size_t k = j; k < vecS; k += 2 * grdS) { */
    /*     local_val = dev_functor::eval(local_val, ev.r.eval(k)); */
    /*     if (k + blqS < vecS) { */
    /*       local_val = dev_functor::eval(local_val, ev.r.eval(k + blqS)); */
    /*     } */
    /*   } */
    /*   // Reduction inside the block */
    /*   val = dev_functor::eval(val, local_val); */
    /* } */
    /* return ev.l.eval(i) = val; */
    return ev.l.eval(ndItem);
  }
};

template <typename EvaluatorT, typename Scratch>
struct KernelFunctor {
  EvaluatorT ev;
  Scratch scratch;

  KernelFunctor(EvaluatorT &ev, Scratch scratch) : ev(ev), scratch(scratch) {}

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
    return ev.l.eval(groupid);
  }
};

template <class EvaluatorT>
class GenericReducer {
  using Device = typename EvaluatorT::Device;

  void execute(Device &dev, EvaluatorT ev, size_t localsize,
               size_t globalsize) {
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      h.require(*ev.result, nTree.result);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration, KernelFunctorNoScratch<EvaluatorT>(ev));
    });
  }

  template <typename Scratch>
  void execute(Device &dev, EvaluatorT ev, size_t localsize, size_t globalsize,
               size_t sharedsize) {
    using value_type = typename EvaluatorT::value_type;
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      auto scratch = shared_mem<value_type>(sharedsize, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration,
                     KernelFunctor<EvaluatorT, Scratch>(ev, scratch));
    });
  }
};

template <class EvaluatorT>
class GenericReducerTwoStage : GenericReducer<EvaluatorT> {
  using ExpressionT = typename EvaluatorT::Expression;
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;
  static void run(ExpressionT expr, Device &dev) {
    /* Evaluator<ExpressionT, Device> ev(expr); */
    /* // first stage, N -> nWG */
    /* size_t localsize = 256; */
    /* size_t nwg = 512; */
    /* size_t sharedsize = nwg; */
    /* size_t globalsize = ev.getSize(); */
    /* /1* shared_mem<value_type> shMem(sharedsize); *1/ */
    /* KernelFunctor<decltype(shMem)>(ev, shMem).execute(localsize, localsize *
     * nwg); */
    /* KernelFunctorNoScratch(ev).execute(localsize, nwg); */
  }
};

template <class EvaluatorT>
class GenericReducerClassic : GenericReducer<EvaluatorT> {
  using Device = typename EvaluatorT::Device;
  using value_type = typename EvaluatorT::value_type;
  using oper_type = typename Converter<EvaluatorT>::oper_type;
  using dev_functor = typename EvaluatorT::dev_functor;
  using input_type = typename Converter<EvaluatorT>::input_type;
  using cont_type = typename Converter<EvaluatorT>::cont_type;
  using LHS_type = typename Converter<EvaluatorT>::LHS_type;

  static void run(EvaluatorT &ev, Device &dev) {
    // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF ELEMENTS
    // THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
    // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
    // ALL THE ELEMENTS ARE PROCESSED
    /* EvaluatorT ev(expr); */
    /* size_t N = ev.getSize(); */
    /* size_t localsize = 256; */
    /* size_t nwg = (512 + (2 * localsize) - 1) / (2 * localsize); */
    /* size_t sharedsize = std::min(nwg, localsize); */
    /* auto lhs = ev.l; */
    /* auto rhs = ev.r; */

    /* // Two accessors to local memory */
    /* cl::sycl::buffer<value_type, 1> shMem1(sharedSize); */
    /* cl::sycl::buffer<value_type, 1> shMem2(sharedSize); */

    /* bool frst = true; */
    /* bool even = false; */
    /* KernelFunctor<decltype(shMem1)> kernel_functor_shm1(ev, shMem1); */
    /* KernelFunctor<decltype(shMem2)> kernel_functor_shm2(ev, shMem2); */
    /* KernelFunctorNoScratch kernel_functor_noscratch(ev); */
    /* do { */
    /*   size_t globalsize = nWG * localsize; */
    /*   if (frst) { */
    /*     if (nwg == 1) { */
    /*       kernel_functor_shm1.execute(dev, localsize, globalsize,
     * sharedsize); */
    /*     } else { */
    /*       kernel_functor_shm2.execute(dev, localsize, globalsize, 1); */
    /*     } */
    /*     /1* // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION *1/
     */
    /*     /1* auto localEval = ExpressionT(((nWG == 1) ? lhs : shMemEval1),
     * rhs); */
    /*      *1/ */
    /*     /1* execute_subtree<Device,
     * using_shared_mem::enabled>(dev.sycl_queue(), */
    /*      * localEval, localsize, globalsize, sharedSize); *1/ */
    /*   } else { */
    /*     if (nwg == 1) { */
    /*       kernel_functor_noscratch.execute(dev, localsize, globalsize); */
    /*     } else { */
    /*       if (!even) { */
    /*         kernel_functor_shm1.execute(dev, localsize, globalsize,
     * sharedsize); */
    /*       } else { */
    /*         kernel_functor_shm2.execute(dev, localsize, globalsize,
     * sharedsize); */
    /*       } */
    /*     } */
    /*     /1* // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION *1/ */
    /*     /1* auto localEval = (nWG == 1) ? lhs : (even ? shMemEval2 :
     * shMemEval1), */
    /*      * (even ? shMemEval1 : shMemEval2); *1/ */
    /*     /1* execute_subtree<Device,
     * using_shared_mem::enabled>(dev.sycl_queue(), */
    /*      * localEval, localsize, globalsize, sharedSize); *1/ */
    /*   } */
    /*   N = nWG; */
    /*   nWG = (N + (2 * localsize) - 1) / (2 * localsize); */
    /*   frst = false; */
    /*   even = !even; */
    /* } while (N > 1); */
  }
};

template <class EvaluatorT, class Reducer>
struct FullReducer {
  using Device = typename EvaluatorT::Device;
  static void run(EvaluatorT &ev, Device &dev) {
    /* Reducer::reduce(ev, dev, result); */
  }
};

template <class EvaluatorT, class Reducer>
struct PartialReducer {
  using Device = typename EvaluatorT::Device;
  static void run(EvaluatorT &expr, Device &dev) {}
};

/*!
 * @brief Applies a reduction to a tree.
 */
/* template <typename Functor, typename LHS, typename RHS> */
/* void Evaluator<ReductionExpr<Functor, LHS, RHS>,
 * SYCLDevice>::reduce(SYCLDevice &dev) { */
/*   using Expression = ReductionExpr<Functor, LHS, RHS>; */
/*   using EvaluatorT = Evaluator<Expression, SYCLDevice>; */
/*   using oper_type = typename Converter<Evaluator<Expression,
 * Device>>::oper_type; */
/*   using input_type = typename Converter<Evaluator<Expression,
 * Device>>::input_type; */
/*   using cont_type = typename Converter<Evaluator<Expression,
 * Device>>::cont_type; */
/*   using LHS_type = typename Converter<Evaluator<Expression,
 * Device>>::LHS_type; */
/*   auto _N = getSize(); */
/*   auto localsize = 256; */
/*   // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF ELEMENTS
 */
/*   // THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP */
/*   // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL */
/*   // ALL THE ELEMENTS ARE PROCESSED */
/*   auto nWG = (512 + (2 * localsize) - 1) / (2 * localsize); */
/*   auto lhs = this->l; */
/*   auto rhs = this->r; */

/*   // Two accessors to local memory */
/*   auto sharedSize = ((nWG < localsize) ? localsize : nWG); */
/*   cl::sycl::buffer<typename EvaluatorT::value_type, 1> shMem1(sharedSize); */
/*   cl::sycl::buffer<typename EvaluatorT::value_type, 1> shMem2(sharedSize); */

/*   bool frst = true; */
/*   bool even = false; */
/*   do { */
/*     auto globalsize = nWG * localsize; */
/*     if(frst) { */
/*       // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION */
/*       auto localEval = Expression(((nWG == 1) ? lhs : shMemEval1), rhs); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(),
 * localEval, localsize, globalsize, sharedSize); */
/*     } else { */
/*       // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION */
/*       auto localEval = (nWG == 1) ? lhs : (even ? shMemEval2 : shMemEval1),
 * (even ? shMemEval1 : shMemEval2); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(),
 * localEval, localsize, globalsize, sharedSize); */
/*     } */
/*     _N = nWG; */
/*     nWG = (_N + (2 * localsize) - 1) / (2 * localsize); */
/*     frst = false; */
/*     even = !even; */
/*   } while (_N > 1); */
/* }; */

/*!
 * @brief Applies a reduction to a tree, receiving a scratch buffer.
 */
/* template <typename EvaluatorT, typename Scratch> */
/* void reduce(SYCLDevice &dev, EvaluatorT ev, Scratch scr) { */
/*   using oper_type = typename blas::Converter<Evaluator<ExpressionT,
 * Device>>::oper_type; */
/*   using input_type = typename blas::Converter<Evaluator<ExpressionT,
 * Device>>::input_type; */
/*   using cont_type = typename blas::Converter<Evaluator<ExpressionT,
 * Device>>::cont_type; */
/*   using LHS_type = typename blas::Converter<Evaluator<ExpressionT,
 * Device>>::LHS_type; */
/*   auto _N = ev.getSize(); */
/*   auto localsize = ev.blqS; */
/*   // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF ELEMENTS
 */
/*   // THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP */
/*   // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL */
/*   // ALL THE ELEMENTS ARE PROCESSED */
/*   auto nWG = (ev.grdS + (2 * localsize) - 1) / (2 * localsize); */
/*   auto lhs = ev.l; */
/*   auto rhs = ev.r; */

/*   // Two accessors to local memory */
/*   auto sharedSize = ((nWG < localsize) ? localsize : nWG); */
/*   auto opShMem1 = LHS_type(scr, 0, 1, sharedSize); */
/*   auto opShMem2 = LHS_type(scr, sharedSize, 1, sharedSize); */

/*   bool frst = true; */
/*   bool even = false; */
/*   do { */
/*     auto globalsize = nWG * localsize; */
/*     if (frst) { */
/*       // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION */
/*       auto localEval = input_type(((nWG == 1) ? lhs : opShMem1), rhs,
 * localsize, globalsize); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(),
 * localExpr, localsize, */
/*                                               globalsize, sharedSize); */
/*     } else { */
/*       // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION */
/*       auto localEval = Evaluator<ReductionExpr<oper_type, LHS_type,
 * LHS_type>, Device>( */
/*               ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)), */
/*               (even ? opShMem1 : opShMem2), localsize, globalsize); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(),
 * localExpr, localsize, globalsize, sharedSize); */
/*     } */
/*     _N = nWG; */
/*     nWG = (_N + (2 * localsize) - 1) / (2 * localsize); */
/*     frst = false; */
/*     even = !even; */
/*   } while (_N > 1); */
/* } */

}  // namespace BLAS

#endif
