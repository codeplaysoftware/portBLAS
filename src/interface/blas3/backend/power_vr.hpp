/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
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
 *  @filename power_vr.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_POWERVR_BACKEND_HPP
#define SYCL_BLAS_GEMM_POWERVR_BACKEND_HPP
#include "interface/gemm_launcher.h"

#ifdef IMGDNN_LIBRARY
#include <SYCL/codeplay.hpp>
#include <imgdnn/cl.h>
#include <imgdnn/imgdnn.h>
#include <iostream>
#endif

namespace blas {
namespace gemm {
namespace backend {

#ifdef IMGDNN_LIBRARY
namespace sycl_imagination_nn_api {
/*!
 * @brief Select the correct transpose version of GemmFactory, depending on the
 *        runtime values of transpose.
 */
template <bool TransA, bool TransB>
struct Gemm_Launcher {
  template <typename sb_handle_t, typename container_0_t,
            typename container_1_t, typename container_2_t, typename value_t,
            typename index_t>
  static inline typename sb_handle_t::event_t _select_gemm(
      sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
      value_t _alpha, container_0_t _A, container_1_t _B, value_t _beta,
      container_2_t _C, index_t batch_size) {
    auto m = static_cast<size_t>(_M);
    auto n = static_cast<size_t>(_N);
    auto k = static_cast<size_t>(_K);
    auto n_batches = static_cast<size_t>(batch_size);
    cl::sycl::event sycl_event;
    // we swap the matrix as they are row major while the netlib blas require
    // column major so C := alpha x (A * B) + (beta x C) will be equal to C :=
    // alphax (B * A) + (beta x C)
    auto a_buffer = _A.get_buffer();
    auto b_buffer = _B.get_buffer();
    auto c_buffer = _C.get_buffer();
    auto interop_event = sb_handle.get_queue().submit([&](cl::sycl::codeplay::
                                                              handler& cgh) {
      auto a_acc =
          a_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto b_acc =
          b_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto c_acc =
          c_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.interop_task([&, a_acc, b_acc, c_acc](
                           const cl::sycl::codeplay::interop_handle& handle) {
        auto m_cl_device = handle.get_device();
        auto m_cl_context = handle.get_context();
        imgdnn_device device_;
        imgdnn_err_code imgdnn_err = IMGDNN_SUCCESS;
        auto context_ = imgdnnCLCreateContext(m_cl_context, 1u, &m_cl_device,
                                              0u, &device_, &imgdnn_err);

        auto _A_cl_mem_object = imgdnnImportMemory(
            context_, handle.get(a_acc),
            m * k * sizeof(typename blas::ValueType<container_0_t>::type),
            IMGDNN_IMPORT_MEM_TYPE_OPENCL, nullptr);

        auto _B_cl_mem_object = imgdnnImportMemory(
            context_, handle.get(b_acc),
            n * k * sizeof(typename blas::ValueType<container_1_t>::type),
            IMGDNN_IMPORT_MEM_TYPE_OPENCL, nullptr);

        auto _C_cl_mem_object = imgdnnImportMemory(
            context_, handle.get(c_acc),
            m * n * sizeof(typename blas::ValueType<container_2_t>::type),
            IMGDNN_IMPORT_MEM_TYPE_OPENCL, nullptr);

        imgdnn_tensor_descriptor lhs_descriptor, rhs_descriptor, c_descriptor;

        if (TransB) {
          lhs_descriptor = {
              .dimensions = 3,
              .type = IMGDNN_TYPE_F32,
              .size = {n_batches, k, n},
          };
        } else {
          lhs_descriptor = {
              .dimensions = 3,
              .type = IMGDNN_TYPE_F32,
              .size = {n_batches, n, k},
          };
        }
        if (TransA) {
          rhs_descriptor = {
              .dimensions = 3,
              .type = IMGDNN_TYPE_F32,
              .size = {n_batches, m, k},
          };
        } else {
          rhs_descriptor = {
              .dimensions = 3,
              .type = IMGDNN_TYPE_F32,
              .size = {n_batches, k, m},
          };
        }

        c_descriptor = {
            .dimensions = 3,
            .type = IMGDNN_TYPE_F32,
            .size = {n_batches, n, m},
        };

        imgdnn_tensor_descriptor scaling_tensor_descriptor = {
            .dimensions = 3, .type = IMGDNN_TYPE_F32, .size = {1, 1, 1}};

        auto network = imgdnnCreateNetwork(nullptr);
        auto binding = imgdnnCreateBinding(nullptr);
        auto lhs_input =
            imgdnnNetworkInput(network, &lhs_descriptor, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE third input: " << imgdnn_err << std::endl;
#endif
        auto rhs_input =
            imgdnnNetworkInput(network, &rhs_descriptor, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE third input: " << imgdnn_err << std::endl;
#endif

        auto c_input = imgdnnNetworkInput(network, &c_descriptor, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE third input: " << imgdnn_err << std::endl;
#endif
        imgdnn_tensor net_inputs[3] = {lhs_input, rhs_input, c_input};

        const int new_order[3] = {0, 2, 1};

        imgdnn_tensor matmul_inputs[2];
        matmul_inputs[0] = (TransB)
                               ? imgdnnNetworkTransposeOp(
                                     network, lhs_input, new_order, &imgdnn_err)
                               : lhs_input;
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE transpose B: " << imgdnn_err << std::endl;
#endif
        matmul_inputs[1] = (TransA)
                               ? imgdnnNetworkTransposeOp(
                                     network, rhs_input, new_order, &imgdnn_err)
                               : rhs_input;
#ifdef BLAS_VERBOSE

        std::cout << "ERROR CODE transpose A: " << imgdnn_err << std::endl;
#endif
        auto A_Times_B_result =
            imgdnnNetworkBinaryOp(network, matmul_inputs[0], matmul_inputs[1],
                                  IMGDNN_OPERATION_MATMUL, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE a times b: " << imgdnn_err << std::endl;
#endif
        auto alpha_scaling = imgdnnNetworkFixedInput(
            network, &scaling_tensor_descriptor, &_alpha, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE fix_input: " << imgdnn_err << std::endl;
#endif
        auto matmul_result =
            imgdnnNetworkBinaryOp(network, alpha_scaling, A_Times_B_result,
                                  IMGDNN_OPERATION_MUL, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE create alpha: " << imgdnn_err << std::endl;
#endif

        imgdnn_tensor output_tensor = matmul_result;

        if (_beta != 0.f) {
          auto beta_scaling = imgdnnNetworkFixedInput(
              network, &scaling_tensor_descriptor, &_beta, &imgdnn_err);
#ifdef BLAS_VERBOSE
          std::cout << "ERROR CODE create bets: " << imgdnn_err << std::endl;
#endif
          imgdnn_tensor beta_scaled =
              imgdnnNetworkBinaryOp(network, beta_scaling, c_input,
                                    IMGDNN_OPERATION_MUL, &imgdnn_err);
#ifdef BLAS_VERBOSE
          std::cout << "ERROR CODE res times beta: " << imgdnn_err << std::endl;
#endif
          output_tensor =
              imgdnnNetworkBinaryOp(network, beta_scaled, matmul_result,
                                    IMGDNN_OPERATION_ADD, &imgdnn_err);
#ifdef BLAS_VERBOSE
          std::cout << "ERROR CODE a times b plus c: " << imgdnn_err
                    << std::endl;
#endif
        }
        auto network_object = imgdnnCreateNetworkObject(
            device_, context_, network, 3u, net_inputs, 1u, &output_tensor, 0u,
            nullptr, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  create network object: " << imgdnn_err
                  << std::endl;
#endif
        imgdnn_input inputs[3];
        imgdnn_err =
            imgdnnNetworkObjectGetInputs(network_object, 3u, inputs, nullptr);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  get input objects: " << imgdnn_err
                  << std::endl;
#endif
        imgdnn_output output;
        imgdnn_err =
            imgdnnNetworkObjectGetOutputs(network_object, 1u, &output, nullptr);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  get output objects: " << imgdnn_err
                  << std::endl;
#endif
        // swap the inputs
        imgdnn_err =
            imgdnnBindingAddInput(binding, inputs[0], _B_cl_mem_object);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  bind input object 0: " << imgdnn_err
                  << std::endl;
#endif
        // swap the inputs
        imgdnn_err =
            imgdnnBindingAddInput(binding, inputs[1], _A_cl_mem_object);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  bind input object 1: " << imgdnn_err
                  << std::endl;
#endif
        imgdnn_err =
            imgdnnBindingAddInput(binding, inputs[2], _C_cl_mem_object);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  bind input object 2: " << imgdnn_err
                  << std::endl;
#endif
        imgdnn_err = imgdnnBindingAddOutput(binding, output, _C_cl_mem_object);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  bind output object: " << imgdnn_err
                  << std::endl;
#endif
        imgdnn_event ev;
        imgdnn_err = imgdnnNetworkObjectExecute(network_object, binding, true,
                                                0u, nullptr, &ev);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  get event object: " << imgdnn_err;
#endif

        cl_event cl_ev = imgdnnCLExportEvent(ev, &imgdnn_err);
#ifdef BLAS_VERBOSE
        std::cout << "ERROR CODE  export cl event object: " << imgdnn_err
                  << std::endl;
        sycl_event = cl::sycl::event{cl_ev, m_cl_context};
#endif
        imgdnnBindingDestroy(binding);
        imgdnnNetworkObjectDestroy(network_object);
        imgdnnNetworkDestroy(network);
        imgdnnEventDestroy(ev);
        imgdnnMemoryDestroy(_A_cl_mem_object);
        imgdnnMemoryDestroy(_B_cl_mem_object);
        imgdnnMemoryDestroy(_C_cl_mem_object);
        imgdnnContextDestroy(context_);
        clReleaseEvent(cl_ev);
      });
    });
    interop_event.wait();
    return {sycl_event};
  }
};

}  // namespace sycl_imagination_nn_api

#endif

template <bool _t_a, bool _t_b, bool is_beta_zero, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(sb_handle_t& sb_handle, index_t _M,
                                    index_t _N, index_t _K, element_t _alpha,
                                    container_0_t _a, index_t _lda,
                                    container_1_t _b, index_t _ldb,
                                    element_t _beta, container_2_t _c,
                                    index_t _ldc, index_t batch_size,
                                    gemm_batch_type_t batch_type) {
#ifdef IMGDNN_LIBRARY
  if (batch_type == gemm_batch_type_t::interleaved) {
    std::cerr << "Error: interleaved gemm is not supported with IMGDNN"
              << std::endl;
    return {};
  }
  return blas::gemm::backend::sycl_imagination_nn_api::Gemm_Launcher<
      _t_a, _t_b>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _b,
                                         _beta, _c, batch_size);
#else
  if (batch_type == gemm_batch_type_t::interleaved) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 4, 4, 1, 1, 1, 1, 4, 4>, _t_a,
        _t_b, static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::interleaved)>::template _select_gemm(sb_handle,
                                                                    _M, _N, _K,
                                                                    _alpha, _a,
                                                                    _lda, _b,
                                                                    _ldb, _beta,
                                                                    _c, _ldc,
                                                                    batch_size);
  }
  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  if ((_M == 96 && _K == 16 && _N == 22500) ||
      (_M == 273 && _K == 576 && _N == 100) ||
      (_M == 384 && _K == 64 && _N == 361)) {
    return blas::Gemm_Launcher<
        96, true, false, false, 16, Tile<4, 6, 12, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 546 && _K == 512 && _N == 4) ||
           (_M == 24 && _K == 512 && _N == 4) ||
           (_M == 24 && _K == 256 && _N == 1) ||
           (_M == 64 && _K == 256 && _N == 4) ||
           (_M == 24 && _K == 256 && _N == 1) ||
           (_M == 128 && _K == 64 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 128, Tile<1, 1, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 546 && _K == 128 && _N == 1) ||
           (_M == 546 && _K == 256 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 576 && _K == 96 && _N == 361) ||
           (_M == 64 && _K == 384 && _N == 361) ||
           (_M == 160 && _K == 576 && _N == 100) ||
           (_M == 1280 && _K == 320 && _N == 100) ||
           (_M == 256 && _K == 1280 && _N == 100) ||
           (_M == 960 && _K == 160 && _N == 100) ||
           (_M == 192 && _K == 32 && _N == 1444) ||
           (_M > 64 && _K > 64 && _N > 64 && is_power_of_2(_M) &&
            is_power_of_2(_K) && is_power_of_2(_N))) {
    return blas::Gemm_Launcher<
        128, false, false, false, 16, Tile<4, 8, 16, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  } else {
    return blas::Gemm_Launcher<
        64, false, false, false, 32, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  }
#endif
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
