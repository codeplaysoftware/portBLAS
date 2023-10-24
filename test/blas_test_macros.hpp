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
 *  portBLAS: BLAS implementation using SYCL
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
#define BLAS_REGISTER_TEST_FLOAT_CUSTOM_NAME(test_suite, class_name,          \
                                             test_function, combination_t,    \
                                             combination, name_generator)     \
  class class_name##Float                                                     \
      : public ::testing::TestWithParam<combination_t<float>> {};             \
  TEST_P(class_name##Float, test) { test_function<float>(GetParam()); };      \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Float, combination<float>, \
                           name_generator<float>);

#ifdef BLAS_DATA_TYPE_DOUBLE
/** Registers test for the double type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_DOUBLE_CUSTOM_NAME(test_suite, class_name,       \
                                              test_function, combination_t, \
                                              combination, name_generator)  \
  class class_name##Double                                                  \
      : public ::testing::TestWithParam<combination_t<double>> {};          \
  TEST_P(class_name##Double, test) { test_function<double>(GetParam()); };  \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Double,                  \
                           combination<double>, name_generator<double>);
#else
#define BLAS_REGISTER_TEST_DOUBLE_CUSTOM_NAME(test_suite, class_name,       \
                                              test_function, combination_t, \
                                              combination, name_generator)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
/** Registers test for the cl::sycl::half type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_HALF_CUSTOM_NAME(test_suite, class_name,        \
                                            test_function, combination_t,  \
                                            combination, name_generator)   \
  class class_name##Half                                                   \
      : public ::testing::TestWithParam<combination_t<cl::sycl::half>> {}; \
  TEST_P(class_name##Half, test) {                                         \
    test_function<cl::sycl::half>(GetParam());                             \
  };                                                                       \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##Half,                   \
                           combination<cl::sycl::half>,                    \
                           name_generator<cl::sycl::half>);
#else
#define BLAS_REGISTER_TEST_HALF_CUSTOM_NAME(test_suite, class_name,       \
                                            test_function, combination_t, \
                                            combination, name_generator)
#endif  // BLAS_DATA_TYPE_HALF

#ifdef BLAS_ENABLE_COMPLEX
#define BLAS_REGISTER_TEST_CPLX_S_CUSTOM_NAME(test_suite, class_name,        \
                                              test_function, combination_t,  \
                                              combination, name_generator)   \
  class class_name##CplxFloat                                                \
      : public ::testing::TestWithParam<combination_t<float>> {};            \
  TEST_P(class_name##CplxFloat, test) { test_function<float>(GetParam()); }; \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##CplxFloat,                \
                           combination<float>, name_generator<float>);
#else
#define BLAS_REGISTER_TEST_CPLX_S_CUSTOM_NAME(test_suite, class_name,       \
                                              test_function, combination_t, \
                                              combination, name_generator)
#endif  // BLAS_ENABLE_COMPLEX

#if defined(BLAS_DATA_TYPE_DOUBLE) & defined(BLAS_ENABLE_COMPLEX)
#define BLAS_REGISTER_TEST_CPLX_D_CUSTOM_NAME(test_suite, class_name,          \
                                              test_function, combination_t,    \
                                              combination, name_generator)     \
  class class_name##CplxDouble                                                 \
      : public ::testing::TestWithParam<combination_t<double>> {};             \
  TEST_P(class_name##CplxDouble, test) { test_function<double>(GetParam()); }; \
  INSTANTIATE_TEST_SUITE_P(test_suite, class_name##CplxDouble,                 \
                           combination<double>, name_generator<double>);
#else
#define BLAS_REGISTER_TEST_CPLX_D_CUSTOM_NAME(test_suite, class_name,       \
                                              test_function, combination_t, \
                                              combination, name_generator)
#endif  // BLAS_ENABLE_COMPLEX & BLAS_ENABLE_COMPLEX

/** Registers test for all supported data types
 * @param test_suite Name of the test suite
 * @param class_name Base name of the test class
 * @param test_function Templated function used to run the test
 * @param combination_t Type of the combination parameter.
 *        Must be templated for the data type.
 * @param combination Combinations object
 * @param name_generator Function used to generate test names
 */
#define BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, class_name, test_function,  \
                                       combination_t, combination,             \
                                       name_generator)                         \
  BLAS_REGISTER_TEST_FLOAT_CUSTOM_NAME(test_suite, class_name, test_function,  \
                                       combination_t, combination,             \
                                       name_generator);                        \
  BLAS_REGISTER_TEST_DOUBLE_CUSTOM_NAME(test_suite, class_name, test_function, \
                                        combination_t, combination,            \
                                        name_generator);                       \
  BLAS_REGISTER_TEST_HALF_CUSTOM_NAME(test_suite, class_name, test_function,   \
                                      combination_t, combination,              \
                                      name_generator);

#ifdef BLAS_ENABLE_COMPLEX
#define BLAS_REGISTER_CPLX_TEST_CUSTOM_NAME(test_suite, class_name,            \
                                            test_function, combination_t,      \
                                            combination, name_generator)       \
  BLAS_REGISTER_TEST_CPLX_S_CUSTOM_NAME(test_suite, class_name, test_function, \
                                        combination_t, combination,            \
                                        name_generator);                       \
  BLAS_REGISTER_TEST_CPLX_D_CUSTOM_NAME(test_suite, class_name, test_function, \
                                        combination_t, combination,            \
                                        name_generator);
#endif  // BLAS_ENABLE_COMPLEX

/** Registers test for all supported data types
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_ALL(class_name, combination_t, combination, \
                               name_generator)                         \
  BLAS_REGISTER_TEST_CUSTOM_NAME(class_name, class_name, run_test,     \
                                 combination_t, combination, name_generator)

/** Registers test for the float data type
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define BLAS_REGISTER_TEST_FLOAT(class_name, combination_t, combination, \
                                 name_generator)                         \
  BLAS_REGISTER_TEST_FLOAT_CUSTOM_NAME(class_name, class_name, run_test, \
                                       combination_t, combination,       \
                                       name_generator)

#endif  // TEST_BLAS_TEST_MACROS_HPP
