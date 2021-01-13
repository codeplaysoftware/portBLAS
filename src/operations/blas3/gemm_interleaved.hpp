/***************************************************************************
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
 *  @filename gemm_interleaved.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_INTERLEAVED_HPP
#define SYCL_BLAS_BLAS3_GEMM_INTERLEAVED_HPP

#include "gemm_common.hpp"

namespace blas {

namespace internal {

template <class T, int Dim>
struct packet {
  using type = cl::sycl::vec<T, Dim>;
};

template <class T>
struct packet<T, 1> {
  using type = T;
};

using address_t = cl::sycl::access::address_space;

/*!
 * @brief Load a packet of size 1.
 */
template <address_t Address = address_t::global_space, class T, class PtrT>
SYCL_BLAS_INLINE void load(T &packet, PtrT ptr) {
  packet = *ptr;
}

/*!
 * @brief Store a packet of size 1.
 */
template <address_t Address = address_t::global_space, class T, class PtrT>
SYCL_BLAS_INLINE void store(T packet, PtrT ptr) {
  *ptr = packet;
}

/*!
 * @brief Load a packet of size Dim.
 */
template <address_t Address = address_t::global_space, class T, int Dim,
          class PtrT>
SYCL_BLAS_INLINE void load(cl::sycl::vec<T, Dim> &packet, PtrT ptr) {
  packet.template load<Address>(0, ptr);
}

/*!
 * @brief Store a packet of size Dim.
 */
template <address_t Address = address_t::global_space, class T, int Dim,
          class PtrT>
SYCL_BLAS_INLINE void store(const cl::sycl::vec<T, Dim> &packet, PtrT ptr) {
  packet.template store<Address>(0, ptr);
}

}  // namespace internal

/*!
 * @brief This partial specialization of the Gemm class add supports for
 * the interleaved batch type gemm. The batch is the "fast moving" dimension
 * making it easier to vectorize on the batch dimension.
 * One notable difference with other gemm implementation is that the work
 * groups have to be tiled in the 3 dimensions. Best performance is achieved if
 * the batch dimensions is divisible by the tiled batch dimension.
 * Each workgroup computes
 * (item_mb * wg_batchs * item_rows * wg_rows * item_cols * wg_cols).
 *
 * @tparam input_t  input view type
 * @tparam output_t  output view type
 * @tparam ClSize  unused in this kernel
 * @tparam tile_type  determines the size of the local, work group, and top
 *                    level tiles to use, see Tile
 * @tparam TransA  if true, matrix A will be transposed on the fly
 * @tparam TransB  if true, matrix B will be transposed on the fly
 * @tparam element_t  type of matrix elements
 * @tparam is_beta_zero  whether to optimize away the beta * C addition
 */
template <typename input_t, typename output_t, int ClSize, typename tile_type,
          bool TransA, bool TransB, typename element_t, bool is_beta_zero,
          int VectorSize>
class Gemm<input_t, output_t, /* DoubleBuffer = */ false, /* NbcA = */ false,
           /* NbcB = */ false, ClSize, tile_type, TransA, TransB, element_t,
           is_beta_zero, static_cast<int>(gemm_memory_t::no_local),
           static_cast<int>(gemm_algorithm_t::standard),
           static_cast<int>(gemm_vectorization_t::full), VectorSize,
           static_cast<int>(gemm_batch_type_t::interleaved)> {
 public:
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using address_t = cl::sycl::access::address_space;
  static constexpr int local_memory_size = 0;
  /*! @brief The number of rows processed by each work item */
  static constexpr index_t item_rows = tile_type::item_rows;
  /*! @brief The number of cols processed by each work item */
  static constexpr index_t item_cols = tile_type::item_cols;
  /*! @brief The number of batchs processed by each work item */
  static constexpr index_t item_batchs = tile_type::item_batchs;
  /*! @brief The number of work items in each row of work group */
  static constexpr index_t wg_rows = tile_type::wg_rows;
  /*! @brief The number of work items in each column of work group */
  static constexpr index_t wg_cols = tile_type::wg_cols;
  /*! @brief The number of work items in each batch of work group */
  static constexpr index_t wg_batchs = tile_type::wg_batchs;
  /*! @brief Number of rows within a work-group level tile */
  static constexpr index_t block_rows = wg_rows * item_rows;
  /*! @brief Number of columns within a work-group level tile */
  static constexpr index_t block_cols = wg_cols * item_cols;
  /*! @brief Number of batchs within a work-group level tile */
  static constexpr index_t block_batchs = wg_batchs;
  /*! @brief A boolean parameter represents whether or not matrix A is
   * transposed */
  static constexpr bool trans_a = TransA;
  /*! @brief A boolean parameter represents whether or not matrix B is
   * transposed */
  static constexpr bool trans_b = TransB;
  /*! @brief The packet type depends on the size of item_batchs as the kernel
   * is vectorizing over the batch dimension */
  using packet_type = typename internal::packet<value_t, VectorSize>::type;

  static_assert(item_batchs % VectorSize == 0,
                "Item batch must be divisible by vector size");

  input_t a_;
  input_t b_;
  output_t c_;
  const element_t alpha_;
  const element_t beta_;
  const index_t m_;
  const index_t n_;
  index_t k_;
  const index_t lda_;
  const index_t ldb_;
  const index_t ldc_;
  const index_t batch_size_;
  SYCL_BLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta / alpha),
        m_(A.get_size_row()),
        n_(B.get_size_col()),
        k_(A.get_size_col()),
        lda_(A.getSizeL()),
        ldb_(B.getSizeL()),
        ldc_(C.getSizeL()),
        batch_size_(batch_size) {}

  /*!
   * @brief Get the type of this Gemm as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "Gemm <" << false << ", " << false << ", " << false << ", " << ClSize
        << ", " << tile_type::get_type_string() << ", "
        << type_string<value_t>::get_value() << "gemm_memory:no_local, "
        << "gemm_algorithm:standard, "
        << "gemm_vectorization:full, "
        << "vector size" << VectorSize << ", batch_type:interleaved>";
    return str.str();
  }

  SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t compute_units) const noexcept {
    const index_t number_of_block_per_row = ((m_ - 1) / block_rows) + 1;
    const index_t number_of_block_per_cols = ((n_ - 1) / block_cols) + 1;

    const index_t number_of_block_per_batch =
        ((batch_size_ - 1) / (block_batchs * item_batchs)) + 1;

    const cl::sycl::range<1> nwg(number_of_block_per_row *
                                 number_of_block_per_cols *
                                 number_of_block_per_batch);
    const cl::sycl::range<1> wgs(wg_rows * wg_cols * wg_batchs);

    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  SYCL_BLAS_INLINE bool valid_thread(const cl::sycl::nd_item<1> &ndItem) const {
    return true;
  }

  SYCL_BLAS_INLINE void eval(cl::sycl::nd_item<1> id) noexcept {
    auto A = a_.get_pointer();
    auto B = b_.get_pointer();
    auto C = c_.get_pointer();

    const index_t number_of_block_per_row = ((m_ - 1) / block_rows) + 1;
    const index_t number_of_block_per_batch =
        ((batch_size_ - 1) / (block_batchs * item_batchs)) + 1;

    /* linear work group id The number of work-group required to executed each
     * batch efficiently */
    const index_t wg_id = id.get_group(0);
    /* linear work item id */
    const index_t item_id = id.get_local_id(0);

    const index_t tile_id_batch = (wg_id % number_of_block_per_batch);
    const index_t tile_id_row =
        (wg_id / number_of_block_per_batch) % number_of_block_per_row;
    const index_t tile_id_col =
        (wg_id / number_of_block_per_batch) / number_of_block_per_row;

    const index_t wg_batch = tile_id_batch * block_batchs;
    const index_t wg_row = tile_id_row * block_rows;
    const index_t wg_col = tile_id_col * block_cols;

    /* is_internal_block is used to check whether a blocks needs boundary
     * checking */
    const bool is_internal_block =
        ((batch_size_ / item_batchs) - wg_batch >= block_batchs) &&
        (m_ - wg_row >= block_rows) && (n_ - wg_col >= block_cols);

    const index_t local_item_id_batch = (item_id % wg_batchs);
    const index_t local_item_id_row = (item_id / wg_batchs) % wg_rows;
    const index_t local_item_id_col = (item_id / wg_batchs) / wg_rows;
    const index_t mb_start =
        (local_item_id_batch * VectorSize) + (wg_batch * item_batchs);
    const index_t m_start = (local_item_id_row * item_rows) + wg_row;
    const index_t n_start = (local_item_id_col * item_cols) + wg_col;

    /* Exiting from any threads outside of the m_, n_, batch boundary */
    const bool out_of_range =
        (mb_start >= batch_size_) || (m_start >= m_) || (n_start >= n_);
    if (out_of_range) {
      return;
    }

    const index_t m_stride = (trans_a ? lda_ : 1) * batch_size_;
    const index_t n_stride = (trans_b ? 1 : ldb_) * batch_size_;

    // K start is always zero
    A += (m_start * m_stride) + mb_start;
    B += (n_start * n_stride) + mb_start;
    C += mb_start + (m_start * batch_size_) + (n_start * ldc_ * batch_size_);

    // boundary check
    const auto boundary_check =
        [&](const index_t &m_index, const index_t &sz)
            SYCL_BLAS_ALWAYS_INLINE { return m_index < sz; };

    if (is_internal_block) {
      compute_panel<false>(boundary_check, m_stride, n_stride, mb_start,
                           m_start, n_start, A, B, C);
    } else {
      compute_panel<true>(boundary_check, m_stride, n_stride, mb_start, m_start,
                          n_start, A, B, C);
    }
  }

  template <bool need_check_boundary, typename check_t, typename in_ptr_t,
            typename out_ptr_t>
  SYCL_BLAS_INLINE void compute_panel(check_t boundary_check, index_t m_stride,
                                      index_t n_stride, index_t mb_start,
                                      index_t m_start, index_t n_start,
                                      in_ptr_t A, in_ptr_t B, out_ptr_t C) {
    packet_type reg_a[item_rows * item_batchs / VectorSize];
    packet_type reg_b[item_cols * item_batchs / VectorSize];
    packet_type reg_res[item_rows * item_cols * item_batchs / VectorSize];
    scaling_c<need_check_boundary>(boundary_check, m_start, n_start, mb_start,
                                   reg_res, C);
    const index_t stride_k_a = (trans_a ? 1 : lda_) * batch_size_;
    const index_t stride_k_b = (trans_b ? ldb_ : 1) * batch_size_;
    do {
      load<item_rows, need_check_boundary>(boundary_check, m_start, mb_start,
                                           m_, reg_a, A, m_stride);
      load<item_cols, need_check_boundary>(boundary_check, n_start, mb_start,
                                           n_, reg_b, B, n_stride);
      compute_block(reg_a, reg_b, reg_res);
      A += stride_k_a;
      B += stride_k_b;
      --k_;
    } while (k_ > 0);
    store<need_check_boundary>(boundary_check, m_start, n_start, mb_start,
                               reg_res, C);
  }

  template <index_t item_size, bool need_check_boundary, typename check_t,
            typename ptr_t>
  SYCL_BLAS_INLINE typename std::enable_if<!need_check_boundary>::type load(
      check_t boundary_check, index_t index_start, index_t mb_start,
      index_t dim_size, packet_type *reg_res, ptr_t input, index_t stride) {
#pragma unroll
    for (int i = 0; i < item_size; ++i) {
#pragma unroll
      for (int j = 0; j < item_batchs; j += VectorSize) {
        internal::load(*reg_res, input + (j * wg_batchs));
        ++reg_res;
      }
      input += stride;
    }
  }

  template <index_t item_size, bool need_check_boundary, typename check_t,
            typename ptr_t>
  SYCL_BLAS_INLINE typename std::enable_if<need_check_boundary>::type load(
      check_t boundary_check, index_t index_start, index_t mb_start,
      index_t dim_size, packet_type *reg_res, ptr_t input, index_t stride) {
#pragma unroll
    for (int i = 0; i < item_size; ++i) {
      auto in_range = boundary_check(i + index_start, dim_size);
      if (!in_range) {
        *reg_res = packet_type{0};
        input += stride;
        ++reg_res;
        continue;
      }
#pragma unroll
      for (int j = 0; j < item_batchs; j += VectorSize) {
        bool b_range = boundary_check(
            mb_start + mb_start + (j * wg_batchs) + VectorSize - 1,
            batch_size_);
        // This should be avoided as we always tend to have batch size to be
        // multiple of vectorsize
        if (!b_range) {
#pragma unroll
          for (int p = 0; p < VectorSize; ++p) {
            bool is_in =
                boundary_check(mb_start + (j * wg_batchs) + p, batch_size_);
            reinterpret_cast<value_t *>(reg_res)[p] =
                is_in ? input[(j * wg_batchs) + p] : value_t(0);
          }
          ++reg_res;
          continue;
        }
        internal::load(*reg_res, input + (j * wg_batchs));
        ++reg_res;
      }
      input += stride;
    }
  }
  // separating the boundary check from the internal one by using enable_if
  // instead of do_check to have more readability
  template <bool need_check_boundary, typename check_t, typename ptr_t>
  SYCL_BLAS_INLINE typename std::enable_if<need_check_boundary>::type store(
      check_t boundary_check, index_t m_start, index_t n_start,
      index_t mb_start, packet_type *reg_res, ptr_t C) {
#pragma unroll
    for (int i = 0; i < item_cols; ++i) {
      auto output = C;
#pragma unroll
      for (int j = 0; j < item_rows; ++j) {
        bool in_range =
            boundary_check(m_start + j, m_) && boundary_check(n_start + i, n_);
        if (!in_range) {
          ++reg_res;
          continue;
        }
#pragma unroll
        for (int b = 0; b < item_batchs; b += VectorSize) {
          bool b_range = boundary_check(
              mb_start + (b * wg_batchs) + VectorSize - 1, batch_size_);
          *reg_res *= alpha_;
          if (!b_range) {
#pragma unroll
            for (int p = 0; p < VectorSize; ++p) {
              auto is_in =
                  boundary_check(mb_start + (b * wg_batchs) + p, batch_size_);
              if (is_in) {
                output[b * wg_batchs + p] =
                    reinterpret_cast<value_t *>(reg_res)[p];
              }
            }
            ++reg_res;
            continue;
          }
          internal::store(*reg_res, output + (b * wg_batchs));
          ++reg_res;
        }
        output += batch_size_;
      }
      C += (ldc_ * batch_size_);
    }
  }
  // The internal block that does not need any boundary check
  template <bool need_check_boundary, typename check_t, typename ptr_t>
  SYCL_BLAS_INLINE typename std::enable_if<!need_check_boundary>::type store(
      check_t boundary_check, index_t m_start, index_t n_start,
      index_t mb_start, packet_type *reg_res, ptr_t C) {
#pragma unroll
    for (int i = 0; i < item_cols; ++i) {
      auto output = C;
#pragma unroll
      for (int j = 0; j < item_rows; ++j) {
#pragma unroll
        for (int b = 0; b < item_batchs; b += VectorSize) {
          *reg_res *= alpha_;
          internal::store(*reg_res, output + (b * wg_batchs));
          ++reg_res;
        }
        output += batch_size_;
      }
      C += (ldc_ * batch_size_);
    }
  }

  template <bool need_check_boundary, typename check_t, typename ptr_t,
            bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<!beta_zero>::type scaling_c(
      check_t boundary_check, index_t m_start, index_t n_start,
      index_t mb_start, packet_type *reg_res, ptr_t C) {
#pragma unroll
    for (int i = 0; i < item_cols; ++i) {
      auto output = C;
#pragma unroll
      for (int j = 0; j < item_rows; ++j) {
        bool in_range = do_check<need_check_boundary>(
            boundary_check(m_start + j, m_) && boundary_check(n_start + i, n_));
        if (!in_range) {
          ++reg_res;
          continue;
        }
#pragma unroll
        for (int b = 0; b < item_batchs; b += VectorSize) {
          bool b_range = do_check<need_check_boundary>(boundary_check(
              mb_start + (b * wg_batchs) + VectorSize - 1, batch_size_));
          if (!b_range) {
            for (int p = 0; p < VectorSize; ++p) {
              auto is_in = do_check<need_check_boundary>(
                  boundary_check(mb_start + (b * wg_batchs) + p, batch_size_));
              reinterpret_cast<value_t *>(reg_res)[p] =
                  is_in ? value_t{output[b * wg_batchs + p] * beta_}
                        : value_t{0};
            }
            ++reg_res;
            continue;
          }
          internal::load(*reg_res, output + (b * wg_batchs));
          *reg_res *= beta_;
          ++reg_res;
        }
        output += batch_size_;
      }
      C += (ldc_ * batch_size_);
    }
  }

  template <bool need_check_boundary, typename check_t, typename ptr_t,
            bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<beta_zero>::type scaling_c(
      check_t, index_t, index_t, index_t, packet_type *reg_res, ptr_t) {
#pragma unroll
    for (int i = 0; i < item_rows * item_cols * (item_batchs / VectorSize);
         ++i) {
      reg_res[i] = packet_type(0);
    }
  }

  /*!
   * @brief The following function compute the partial GEMM for the input
   * block reg_a and reg_b and add the result to the reg_res
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  2D register array used to store the result C
   */
  SYCL_BLAS_INLINE void compute_block(packet_type *reg_a, packet_type *reg_b,
                                      packet_type *reg_res) noexcept {
#pragma unroll
    for (int i = 0; i < item_cols; ++i) {
#pragma unroll
      for (int j = 0; j < item_rows; ++j) {
#pragma unroll
        for (int b = 0; b < item_batchs / VectorSize; ++b) {
          *reg_res = cl::sycl::mad(reg_a[j * (item_batchs / VectorSize) + b],
                                   reg_b[i * (item_batchs / VectorSize) + b],
                                   *reg_res);
          ++reg_res;
        }
      }
    }
  }

  /*!
   * @brief binding the placeholder accessors to the SYCL command group
   * handler.
   * @param h: SYCL command group handler.
   */
  void bind(cl::sycl::handler &h) {
    a_.bind(h);
    b_.bind(h);
    c_.bind(h);
  }

  void adjust_access_displacement() {
    a_.adjust_access_displacement();
    b_.adjust_access_displacement();
    c_.adjust_access_displacement();
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_GEMM_INTERLEAVED_HPP
