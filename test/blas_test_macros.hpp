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
 *  @filename blas_test_macros.hpp
 *
 **************************************************************************/

#ifndef TEST_BLAS_TEST_MACROS_HPP
#define TEST_BLAS_TEST_MACROS_HPP

#ifdef VERBOSE
#define DEBUG_PRINT(command) command
#else
#define DEBUG_PRINT(command)
#endif /* ifdef VERBOSE */

#ifndef SYCL_DEVICE
#define SYCL_DEVICE_SELECTOR cl::sycl::default_selector
#else
#define PASTER(x, y) x##y
#define EVALUATOR(x, y) PASTER(x, y)
#define SYCL_DEVICE_SELECTOR cl::sycl::EVALUATOR(SYCL_DEVICE, _selector)
#undef PASTER
#undef EVALUATOR
#endif /* ifndef SYCL_DEVICE */

/** Registers test for the float type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_FLOAT(test_suite, class_name, test_function,  \
                                 combination_t, combination)             \
  class class_name##Float                                                \
      : public ::testing::TestWithParam<combination_t<float>> {};        \
  TEST_P(class_name##Float, test) { test_function<float>(GetParam()); }; \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Float, combination);

#ifdef BLAS_DATA_TYPE_DOUBLE
/** Registers test for the double type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_DOUBLE(test_suite, class_name, test_function,   \
                                  combination_t, combination)              \
  class class_name##Double                                                 \
      : public ::testing::TestWithParam<combination_t<double>> {};         \
  TEST_P(class_name##Double, test) { test_function<double>(GetParam()); }; \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Double, combination);
#else
#define BLAS_REGISTER_TEST_DOUBLE(test_suite, class_name, test_function, \
                                  combination_t, combination)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
/** Registers test for the cl::sycl::half type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_HALF(test_suite, class_name, test_function,     \
                                combination_t, combination)                \
  class class_name##Half                                                   \
      : public ::testing::TestWithParam<combination_t<cl::sycl::half>> {}; \
  TEST_P(class_name##Half, test) {                                         \
    test_function<cl::sycl::half>(GetParam());                             \
  };                                                                       \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Half, combination);
#else
#define BLAS_REGISTER_TEST_HALF(test_suite, class_name, test_function, \
                                combination_t, combination)
#endif  // BLAS_DATA_TYPE_HALF

/** Registers test for all supported data types
 * @param test_suite Name of the test suite
 * @param class_name Base name of the test class
 * @param test_function Templated function used to run the test
 * @param combination_t Type of the combination parameter.
 *        Must be templated for the data type.
 * @param combination Combinations object
 */
#define BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, class_name, test_function, \
                                       combination_t, combination)            \
  BLAS_REGISTER_TEST_FLOAT(test_suite, class_name, test_function,             \
                           combination_t, combination);                       \
  BLAS_REGISTER_TEST_DOUBLE(test_suite, class_name, test_function,            \
                            combination_t, combination);                      \
  BLAS_REGISTER_TEST_HALF(test_suite, class_name, test_function,              \
                          combination_t, combination);

/** Registers test for all supported data types
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST(class_name, combination_t, combination) \
  BLAS_REGISTER_TEST_CUSTOM_NAME(class_name, class_name, run_test, \
                                 combination_t, combination)

#endif  // TEST_BLAS_TEST_MACROS_HPP
