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
 *  @filename sycl_buffer_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  int offset;
  std::tie(size, offset) = combi;

  std::vector<scalar_t> vX(size, scalar_t(1));
  fill_random(vX);

  std::vector<scalar_t> vR_gpu(size, scalar_t(10));
  std::vector<scalar_t> vR_cpu(size, scalar_t(10));

  for (int i = offset; i < size; i++) {
    vR_cpu[i - offset] = vX[i];
  }

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto a = blas::make_sycl_iterator_buffer<scalar_t>(vX.data(), size);
  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), (a + offset),
                                          vR_gpu.data(), size - offset);
  sb_handle.wait(event);

  ASSERT_TRUE(utils::compare_vectors(vR_gpu, vR_cpu));
}

template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(100, 102400),  // size
                                      ::testing::Values(0, 25)         // offset
);

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, offset;
  BLAS_GENERATE_NAME(info.param, size, offset);
}

BLAS_REGISTER_TEST_ALL(Buffer, combination_t, combi, generate_name);

template <typename scalar_t, typename index_t>
inline BufferIterator<const scalar_t> func(BufferIterator<const scalar_t> buff,
                                           index_t offset) {
  return buff += offset;
}

template <typename scalar_t>
void run_const_test(const combination_t<scalar_t> combi) {
  int size;
  int offset;
  std::tie(size, offset) = combi;

  std::vector<scalar_t> vX(size, scalar_t(1));
  fill_random(vX);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto a = blas::make_sycl_iterator_buffer<scalar_t>(vX.data(), size);
  BufferIterator<const scalar_t> buff = func<scalar_t>(a, offset);
}

BLAS_REGISTER_TEST_ALL(BufferConst, combination_t, combi, generate_name);
