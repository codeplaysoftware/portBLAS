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
 *  @filename blas1_iamin_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "unittest/blas1/blas1_iaminmax_common.hpp"
#include <limits>

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using tuple_t = IndexValueTuple<int, scalar_t>;

  index_t size;
  index_t incX;
  generation_mode_t mode;
  std::tie(size, incX, mode) = combi;

  const scalar_t max = std::numeric_limits<scalar_t>::max();

  // Input vector
  std::vector<scalar_t> x_v(size * incX);
  populate_data<scalar_t>(mode, max, x_v);
  for (int i = 0; i < size * incX; i++) {
    // There is a bug in Openblas where 0s are not handled correctly
    if (x_v[i] == scalar_t{0.0}) {
      x_v[i] = 1.0;
    }
  }

  // Removes infs from the vector
  std::transform(
      std::begin(x_v), std::end(x_v), std::begin(x_v),
      [](scalar_t v) { return utils::clamp_to_limits<scalar_t>(v); });

  // Output scalar
  tuple_t out_s{0, max};

  // Reference implementation
  int out_cpu_s = reference_blas::iamin(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_v, size * incX);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<tuple_t>(int(1));
  blas::helper::copy_to_device(sb_handle.get_queue(), &out_s, gpu_out_s, 1);

  _iamin(sb_handle, size, gpu_x_v, incX, gpu_out_s);
  auto event =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s, &out_s, 1);
  sb_handle.wait(event);

  using data_tuple_t = IndexValueTuple<int, scalar_t>;
  data_tuple_t out_data_s{out_s.ind, static_cast<scalar_t>(out_s.val)};

  // Validate the result
  ASSERT_EQ(out_cpu_s, out_data_s.ind);
  ASSERT_EQ(x_v[out_data_s.ind * incX], out_data_s.val);
  ASSERT_EQ(x_v[out_cpu_s * incX], out_data_s.val);
}

BLAS_REGISTER_TEST_ALL(Iamin, combination_t, combi, generate_name);
