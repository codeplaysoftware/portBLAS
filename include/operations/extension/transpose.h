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
 *  @filename transpose.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TRANSPOSE_H
#define SYCL_BLAS_EXTENSION_TRANSPOSE_H

#include <CL/sycl.hpp>

#include "container/sycl_iterator.h"

namespace blas {

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
class Transpose {
 public:
  using index_t = typename in_t::index_t;
  using value_t = element_t;
  static constexpr int wg_size_ = Tile_size * Tile_size;
  in_t A_;
  out_t At_;
  index_t lda_;
  index_t ldat_;
  index_t stridea_;
  index_t strideat_;
  index_t N_;
  index_t M_;
  value_t alpha_;
  index_t tile_count_m_;
  index_t tile_count_n_;

  Transpose(in_t& A, index_t& stridea, out_t& At, index_t& strideat,
            value_t& alpha)
      : A_(A),
        At_(At),
        lda_(A_.getSizeL()),
        ldat_(At_.getSizeL()),
        M_(A_.get_size_row()),
        N_(A_.get_size_col()),
        alpha_(alpha),
        tile_count_m_{M_ / Tile_size},
        tile_count_n_{N_ / Tile_size},
        stridea_(stridea),
        strideat_(strideat) {}

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> item) const;
  void bind(cl::sycl::handler& cgh);
  void adjust_access_displacement();
  void eval(cl::sycl::nd_item<1> item);
  template <typename local_memory_t>
  void eval(local_memory_t local_mem, cl::sycl::nd_item<1> id);
};

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t, typename index_t>
Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>
make_transpose(in_t& A, index_t& stridea, out_t& At, index_t& strideat,
               element_t& alpha) {
  return Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>(
      A, stridea, At, strideat, alpha);
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_H