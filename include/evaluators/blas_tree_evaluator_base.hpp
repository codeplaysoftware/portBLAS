#ifndef BLAS_TREE_EVALUATOR_BASE_HPP_L8QC4HLH
#define BLAS_TREE_EVALUATOR_BASE_HPP_L8QC4HLH

#include <evaluators/blas_functor_traits.hpp>
#include <operations/blas_trees.hpp>
#include <views/operview_base.hpp>

namespace blas {
namespace internal {

/*! DetectScalar.
 * @brief Class specialization used to detect scalar values in ScalarOp nodes.
 * When the value is not an integral basic type,
 * it is assumed to be a vector and the first value
 * is used.
 */
template <typename T,
          bool isscal = std::is_same<T, int>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value ||
                        std::is_same<T, std::complex<float>>::value ||
                        std::is_same<T, std::complex<double>>::value>
struct DetectScalar;
template <typename T>
struct DetectScalar<T, true> {
  static T get_scalar(T &scalar) { return scalar; }
};
template <typename T>
struct DetectScalar<T, false> {
  static typename T::value_type get_Scalar(T &opSCL) { return opSCL.eval(0); }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
*/
template <typename T>
auto get_scalar(T &scl) -> decltype(DetectScalar<T>::get_scalar(scl)) {
  return DetectScalar<T>::get_scalar(scl);
}

}  // namespace internal

template <class Expression, typename Device>
struct Evaluator;

}  // namespace blas

#endif
