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
 *  @filename collapse_nested_tuple.cpp
 *
 **************************************************************************/
#include "blas_test.hpp"
#include "sycl_blas.hpp"

// inputs combination
template <typename scalar_t>
using combination_t = std::tuple<int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  const int OFFSET = 5;  // The first tuples are offset by this amount
  int size;
  int factor;
  std::tie(size, factor) = combi;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Input buffer
  auto v_in = std::vector<scalar_t>(size);
  fill_random(v_in);
  // Intermediate buffer
  auto v_int = std::vector<IndexValueTuple<int, scalar_t>>(
      size, IndexValueTuple<int, scalar_t>(scalar_t(), int()));
  // Output buffer
  auto v_out = std::vector<IndexValueTuple<int, scalar_t>>(
      size, IndexValueTuple<int, scalar_t>(scalar_t(), int()));

  // Load v_int with v_in as tuples
  {
    const auto gpu_v_in =
        blas::make_sycl_iterator_buffer<scalar_t>(v_in.data(), size);
    auto gpu_v_in_vv = make_vector_view(gpu_v_in, 1, size);
    auto gpu_v_int =
        blas::make_sycl_iterator_buffer<IndexValueTuple<int, scalar_t>>(
            v_int.data(), size);
    auto gpu_v_int_vv = make_vector_view(gpu_v_int, 1, size);

    auto tuples = make_tuple_op(gpu_v_in_vv);
    auto assign_tuple = make_op<Assign>(gpu_v_int_vv, tuples);
    sb_handle.execute(assign_tuple);
  }

  // Increment the indexes, so they are different to the ones in the next step
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(i, v_int[i].ind);
    v_int[i].ind += OFFSET;
  }

  // And the final tuple and collapse
  {
    auto gpu_v_int =
        blas::make_sycl_iterator_buffer<IndexValueTuple<int, scalar_t>>(
            v_int.data(), size);
    auto gpu_v_int_vv = make_vector_view(gpu_v_int, 1, size);
    auto gpu_v_out =
        blas::make_sycl_iterator_buffer<IndexValueTuple<int, scalar_t>>(
            v_out.data(), size);
    auto gpu_v_out_vv = make_vector_view(gpu_v_out, 1, size);

    auto tuples = make_tuple_op(gpu_v_int_vv);
    auto collapsed =
        make_op<ScalarOp, CollapseIndexTupleOperator>(factor, tuples);
    auto assign_tuple = make_op<Assign>(gpu_v_out_vv, collapsed);
    sb_handle.execute(assign_tuple);
  }

  // Check the result
  for (int i = 0; i < size; i++) {
    int expected = i * factor + (i + OFFSET);
    ASSERT_EQ(expected, v_out[i].ind);
    ASSERT_TRUE(utils::almost_equal(v_out[i].val, v_in[i]));
  }
}
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(16, 1023),  // size
                                      ::testing::Values(0, 1, 5));  // factor

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, factor;
  BLAS_GENERATE_NAME(info.param, size, factor);
}

BLAS_REGISTER_TEST_FLOAT(CollapseNestedTuple, combination_t, combi,
                         generate_name);
