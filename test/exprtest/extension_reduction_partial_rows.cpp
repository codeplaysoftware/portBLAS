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
 *  @filename extension_reduction_partial_rows.cpp
 *
 **************************************************************************/

// TODO: cleanup

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, int>;

const auto combi = ::testing::Combine(::testing::Values(16),    // rows
                                      ::testing::Values(24),    // columns
                                      ::testing::Values(3),     // ld_mul
);

// ---------------------------
// Utility to print matrices
// ---------------------------

struct MatrixPrinter {
  template <typename index_t, typename VectorT>
  static inline void eval(index_t w, index_t h, VectorT v, index_t ld) {
    for (index_t i = 0; i < h; i++) {
      std::cerr << "[";
      for (index_t j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[i + (j * ld)];
      }
      std::cerr << "]\n";
    }
  }
};

using index_t = int;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int rows, columns, ld_mul;
  std::tie(rows, columns, ld_mul) = combi;

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  int ld = rows * ld_mul;

  std::vector<scalar_t> in_m(ld * cols);
  std::vector<scalar_t> out_v_gpu(rows);
  std::vector<scalar_t> out_v_cpu(rows);

  fill_random(in_m);
  fill_random(out_v_gpu);
  std::copy(out_v_gpu.begin(), out_v_gpu.end(), out_v_cpu.begin());

  // Reduce the reference by hand
  for(int i = 0; i < rows; i++) {
    out_v_cpu[i] = 0;
    for(int j = 0; j < rows; j++) {
      out_v_cpu += in_m[ld * j + i];
    }
  }

  {
    auto m_in_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(in_m, ld * cols);
    auto v_out_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(out_v_gpu, rows);
    try {
      // TODO: add operation here
    } catch (cl::sycl::exception &e) {
      std::cerr << "Exception occured:" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }

  std::cerr << "Before reduction: " << std::endl;
  MatrixPrinter::eval(cols, rows, in_m, ld);

  // the matrix is now in tsgf._C
  std::cerr << "Reference reduction: " << std::endl;
  MatrixPrinter::eval(1, rows, out_v_cpu, ld);

  std::cerr << "Our reduction: " << std::endl;
  MatrixPrinter::eval(1, rows, out_v_gpu, ld);

  ASSERT_TRUE(utils::compare_vectors(out_v_gpu, out_v_cpu));
}

class ReductionPartialRowsFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(ReductionPartialRowsFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(reduction, ReductionPartialRowsFloat, combi);

#if DOUBLE_SUPPORT
class ReductionPartialRowsDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(ReductionPartialRowsDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(reduction, ReductionPartialRowsDouble, combi);
#endif
