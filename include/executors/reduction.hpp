#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include <executors/blas_packet_traits_sycl.hpp>

/*! execute_subtree.
@brief Static function for executing a tree in SYCL.
@tparam int usingSharedMem specifying whether shared memory is enabled.
@tparam Tree Type of the tree.
@param q_ SYCL queue.
@param t Tree object.
@param _localSize Local work group size.
@param _globalSize Global work size.
@param _shMem Size in elements of the shared memory (should be zero if
usingSharedMem == false).
*/
template <typename Device, int usingSharedMem, typename EvaluatorT>
static void execute_subtree(cl::sycl::queue &q_, EvaluatorT ev, size_t _localSize, size_t _globalSize, size_t _shMem) {
  using value_type = typename shared_mem_type<usingSharedMem, ExpressionT>::type;

  auto localSize = _localSize;
  auto globalSize = _globalSize;
  auto shMem = _shMem;

  auto cg1 = [=](cl::sycl::handler &h) mutable {
    auto nTree = blas::make_accessor(ev, h);
    auto scratch = shared_mem<value_type, usingSharedMem>(shMem, h);
    cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
    h.parallel_for(gridConfiguration, ExecTreeFunctor<usingSharedMem, decltype(nTree), decltype(scratch), value_type>(scratch, nTree));
  };

  q_.submit(cg1);
}

/*!
 * @brief Applies a reduction to a tree.
 */
template <typename EvaluatorT>
void reduce<EvaluatorT, SYCLDevice>(EvaluatorT ev, SYCLDevice &dev) {
  using oper_type = typename Converter<Evaluator<ExpressionT, Device>>::oper_type;
  using input_type = typename Converter<Evaluator<ExpressionT, Device>>::input_type;
  using cont_type = typename Converter<Evaluator<ExpressionT, Device>>::cont_type;
  using LHS_type = typename Converter<Evaluator<ExpressionT, Device>>::LHS_type;
  auto _N = ev.getSize();
  auto localSize = ev.blqS;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF ELEMENTS
  // THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (ev.grdS + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = ev.l;
  auto rhs = ev.r;

  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  cont_type shMem1(sharedSize);
  ExpressionT shMemExpr1(shMem1, 0, 1, sharedSize);
  auto opShMem1 = LHS_type(Evaluator<ExpressionT, Device>(shMemExpr1));
  cont_type shMem2(sharedSize);
  ExpressionT shMemExpr2(shMem2, 0, 1, sharedSize);
  auto opShMem2 = LHS_type(Evaluator<ExpressionT, Device>(shMemExpr2));

  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localExpr = input_type(((nWG == 1) ? lhs : opShMem1), rhs, localSize, globalSize);
      execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(), Evaluator<ExpressionT>(localExpr), localSize, globalSize, sharedSize);
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localExpr = ReductionExpr<oper_type, LHS_type, LHS_type>(
              ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
              (even ? opShMem1 : opShMem2), localSize, globalSize);
      execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(), Evaluator<ExpressionT>(localExpr), localSize, globalSize, sharedSize);
    }
    _N = nWG;
    nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
    frst = false;
    even = !even;
  } while (_N > 1);
};

/*!
 * @brief Applies a reduction to a tree, receiving a scratch buffer.
 */
/* template <typename EvaluatorT, typename Scratch> */
/* void reduce(SYCLDevice &dev, EvaluatorT ev, Scratch scr) { */
/*   using oper_type = typename blas::Converter<Evaluator<ExpressionT, Device>>::oper_type; */
/*   using input_type = typename blas::Converter<Evaluator<ExpressionT, Device>>::input_type; */
/*   using cont_type = typename blas::Converter<Evaluator<ExpressionT, Device>>::cont_type; */
/*   using LHS_type = typename blas::Converter<Evaluator<ExpressionT, Device>>::LHS_type; */
/*   auto _N = ev.getSize(); */
/*   auto localSize = ev.blqS; */
/*   // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF ELEMENTS */
/*   // THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP */
/*   // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL */
/*   // ALL THE ELEMENTS ARE PROCESSED */
/*   auto nWG = (ev.grdS + (2 * localSize) - 1) / (2 * localSize); */
/*   auto lhs = ev.l; */
/*   auto rhs = ev.r; */

/*   // Two accessors to local memory */
/*   auto sharedSize = ((nWG < localSize) ? localSize : nWG); */
/*   auto opShMem1 = LHS_type(scr, 0, 1, sharedSize); */
/*   auto opShMem2 = LHS_type(scr, sharedSize, 1, sharedSize); */

/*   bool frst = true; */
/*   bool even = false; */
/*   do { */
/*     auto globalSize = nWG * localSize; */
/*     if (frst) { */
/*       // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION */
/*       auto localEval = input_type(((nWG == 1) ? lhs : opShMem1), rhs, localSize, globalSize); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(), localExpr, localSize, */
/*                                               globalSize, sharedSize); */
/*     } else { */
/*       // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION */
/*       auto localEval = Evaluator<ReductionExpr<oper_type, LHS_type, LHS_type>, Device>( */
/*               ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)), */
/*               (even ? opShMem1 : opShMem2), localSize, globalSize); */
/*       execute_subtree<Device, using_shared_mem::enabled>(dev.sycl_queue(), localExpr, localSize, globalSize, sharedSize); */
/*     } */
/*     _N = nWG; */
/*     nWG = (_N + (2 * localSize) - 1) / (2 * localSize); */
/*     frst = false; */
/*     even = !even; */
/*   } while (_N > 1); */
/* } */

#endif
