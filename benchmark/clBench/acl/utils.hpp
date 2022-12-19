#ifndef ACL_UTILS_HPP
#define ACL_UTILS_HPP

#include <CL/cl.h>
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Validate.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/Tensor.h>
#include <clBench/clwrap.hpp>
#include <thread>

#ifdef ACL_BACKEND_NEON
#include <arm_compute/runtime/NEON/NEFunctions.h>
#endif

#include <common/common_utils.hpp>

namespace blas_benchmark {

void create_benchmark(blas_benchmark::Args &args, bool *success);

namespace utils {

template <typename Tensor>
inline void map_if_needed(Tensor &) {}

template <typename Tensor>
inline void unmap_if_needed(Tensor &) {}

inline void map_if_needed(arm_compute::CLTensor &tensor) { tensor.map(true); }

inline void unmap_if_needed(arm_compute::CLTensor &tensor) { tensor.unmap(); }

template <typename tensor_t>
void fill_tensor(tensor_t &tensor, std::vector<float> &src) {
  arm_compute::Window window;
  const arm_compute::TensorShape &shape = tensor.info()->tensor_shape();
  window.use_tensor_dimensions(shape);

  map_if_needed(tensor);

  arm_compute::Iterator it(&tensor, window);

  arm_compute::execute_window_loop(
      window,
      [&](const arm_compute::Coordinates &id) {
        int idx = id[0] * shape[1] + id[1];
        *reinterpret_cast<float *>(it.ptr()) = src[idx];
      },
      it);

  unmap_if_needed(tensor);
}

template <typename tensor_t>
void extract_tensor(tensor_t &tensor, std::vector<float> &dst) {
  arm_compute::Window window;
  const arm_compute::TensorShape &shape = tensor.info()->tensor_shape();
  window.use_tensor_dimensions(shape);

  map_if_needed(tensor);

  arm_compute::Iterator it(&tensor, window);

  arm_compute::execute_window_loop(
      window,
      [&](const arm_compute::Coordinates &id) {
        int idx = id[0] * shape[1] + id[1];
        dst[idx] = *reinterpret_cast<float *>(it.ptr());
      },
      it);

  unmap_if_needed(tensor);
}

/**
 * @fn time_event
 * @brief No event time for ACL. Return 0
 */
template <>
inline double time_event<void *>(void *&e) {
  return double(0);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
