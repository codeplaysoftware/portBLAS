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

using combination_t = std::tuple<int, int>;

template <typename scalar_t>
void run_test(const combination_t combi) {
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
  test_executor_t ex(q);
  auto a = blas::make_sycl_iterator_buffer<scalar_t>(vX.data(), size);
  auto event = ex.get_policy_handler().copy_to_host((a + offset), vR_gpu.data(),
                                                    size - offset);
  ex.get_policy_handler().wait(event);

  ASSERT_TRUE(utils::compare_vectors(vR_gpu, vR_cpu));
}

const auto combi = ::testing::Combine(::testing::Values(100, 102400),  // size
                                      ::testing::Values(0, 25)         // offset
);

class BufferFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(BufferFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(buffer, BufferFloat, combi);

#if DOUBLE_SUPPORT
class BufferDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(BufferDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(buffer, BufferDouble, combi);
#endif

template <typename data_t, typename index_t>
inline BufferIterator<const data_t, blas::codeplay_policy> func(
    BufferIterator<const data_t, blas::codeplay_policy> buff, index_t offset) {
  return buff += offset;
}

template <typename scalar_t>
void run_const_test(const combination_t combi) {
  int size;
  int offset;
  std::tie(size, offset) = combi;

  std::vector<scalar_t> vX(size, scalar_t(1));
  fill_random(vX);

  auto q = make_queue();
  test_executor_t ex(q);
  auto a = blas::make_sycl_iterator_buffer<scalar_t>(vX.data(), size);
  BufferIterator<const scalar_t, blas::codeplay_policy> buff =
      func<scalar_t>(a, offset);
}

class BufferConstFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(BufferConstFloat, test) { run_const_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(buffer, BufferConstFloat, combi);

#if DOUBLE_SUPPORT
class BufferConstDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(BufferConstDouble, test) { run_const_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(buffer, BufferConstDouble, combi);
#endif
