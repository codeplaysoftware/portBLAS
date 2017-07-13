#ifndef BLAS1_TEST_HPP_DFQO1OHP
#define BLAS1_TEST_HPP_DFQO1OHP

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <interface/blas1_interface_sycl.hpp>

using namespace cl::sycl;
using namespace blas;

template <typename ClassName>
struct option_size;
#define RANDOM_SIZE UINT_MAX
#define RANDOM_STRD UINT_MAX
#define REGISTER_SIZE(size, test_name)          \
  template <>                                   \
  struct option_size<class test_name> {         \
    static constexpr const size_t value = size; \
  };
template <typename ClassName>
struct option_strd;
#define REGISTER_STRD(strd, test_name)          \
  template <>                                   \
  struct option_strd<class test_name> {         \
    static constexpr const size_t value = strd; \
  };
template <typename ScalarT, typename ClassName>
struct option_prec;
#define REGISTER_PREC(type, val, test_name)   \
  template <>                                 \
  struct option_prec<type, class test_name> { \
    static constexpr const type value = val;  \
  };

// Wraps the above arguments into one template parameter.
// We will treat template-specialized blas_templ_struct as a single class
template <class ScalarT_, class ExecutorType_>
struct blas_templ_struct {
  using scalar_t = ScalarT_;
  using executor_t = ExecutorType_;
};
// A "using" shortcut for the struct
template <class ScalarT_, class ExecutorType_ = SYCL>
using blas1_test_args = blas_templ_struct<ScalarT_, ExecutorType_>;

// the test class itself
template <class B>
class BLAS1_Test;

template <class ScalarT_, class ExecutorType_>
class BLAS1_Test<blas1_test_args<ScalarT_, ExecutorType_>>
    : public ::testing::Test {
 public:
  using ScalarT = ScalarT_;
  using ExecutorType = ExecutorType_;

  BLAS1_Test() {}

  virtual ~BLAS1_Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static size_t rand_size() {
    size_t ret = rand() >> 5;
    int type_size =
        sizeof(ScalarT) * CHAR_BIT - std::numeric_limits<ScalarT>::digits10 - 2;
    return (ret & (std::numeric_limits<size_t>::max() +
                   (size_t(1) << (type_size - 2)))) +
           1;
  }

  template <typename DataType,
            typename ValueType = typename DataType::value_type>
  static void set_rand(DataType &vec, size_t _N) {
    ValueType left(-1), right(1);
    for (size_t i = 0; i < _N; ++i) {
      vec[i] = ValueType(rand() % int(right - left) * 1000) * .001 - right;
    }
  }

  template <typename DataType,
            typename ValueType = typename DataType::value_type>
  static void print_cont(const DataType &vec, size_t _N,
                         std::string name = "vector") {
    std::cout << name << ": ";
    for (size_t i = 0; i < _N; ++i) std::cout << vec[i] << " ";
    std::cout << std::endl;
  }

  template <typename DataType,
            typename ValueType = typename DataType::value_type>
  static buffer<ValueType, 1> make_buffer(DataType &vec) {
    return buffer<ValueType, 1>(vec.data(), vec.size());
  }

  template <typename ValueType>
  static vector_view<ValueType, buffer<ValueType>> make_vview(
      buffer<ValueType, 1> &buf) {
    return vector_view<ValueType, buffer<ValueType>>(buf);
  }

  template <typename DeviceSelector,
            typename = typename std::enable_if<
                std::is_same<ExecutorType, SYCL>::value>::type>
  static cl::sycl::queue make_queue(DeviceSelector s) {
    return cl::sycl::queue(s, [=](cl::sycl::exception_list eL) {
      try {
        for (auto &e : eL) std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });
  }
};

// it is important that all tests are run with the same test size
// so each time we access this function within the same program, we get the same
// randomly generated size
template <class TestClass>
size_t test_size() {
  static bool first = true;
  static size_t N;
  if (first) {
    first = false;
    N = TestClass::rand_size();
  }
  return N;
}

// getting the stride in the same way as the size above
template <class TestClass>
size_t test_strd() {
  static bool first = true;
  static size_t N;
  if (first) {
    first = false;
    N = ((rand() & 1) * (rand() % 5)) + 1;
  }
  return N;
}

// unpacking the parameters within the test function
// B is blas_templ_struct
// TestClass is BLAS1_Test<B>
// T is default (scalar) type for the test (e.g. float, double)
// C is the container type for the test (e.g. std::vector)
// E is the executor kind for the test (sequential, openmp, sycl)
#define B1_TEST(name) TYPED_TEST(BLAS1_Test, name)
#define UNPACK_PARAM(test_name)                        \
  using ScalarT = typename TypeParam::scalar_t;        \
  using TestClass = BLAS1_Test<TypeParam>;             \
  using ExecutorType = typename TypeParam::executor_t; \
  using test = class test_name;
// TEST_SIZE determines the size based on the suggestion
#define TEST_SIZE                                                     \
  ((option_size<test>::value == RANDOM_SIZE) ? test_size<TestClass>() \
                                             : option_size<test>::value)
#define TEST_STRD                                                     \
  ((option_strd<test>::value == RANDOM_SIZE) ? test_strd<TestClass>() \
                                             : option_strd<test>::value)
// TEST_PREC determines the precision for the test based on the suggestion for
// the type
#define TEST_PREC option_prec<ScalarT, test>::value

#endif
