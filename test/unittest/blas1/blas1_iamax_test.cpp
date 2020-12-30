#include "blas_test.hpp"
#include "unittest/blas1/blas1_iaminmax_common.hpp"
#include <limits>

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using tuple_t = IndexValueTuple<int, scalar_t>;

  int size;
  int incX;
  generation_mode_t mode;
  std::tie(size, incX, mode) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input vector
  std::vector<data_t> x_v(size * incX);
  populate_data<data_t>(mode, 0.0, x_v);

  // This will remove infs from the vector
  std::transform(std::begin(x_v), std::end(x_v), std::begin(x_v),
                 [](data_t v) { return utils::clamp_to_limits<scalar_t>(v); });

  // Output scalar
  tuple_t out_s{0, 0.0};

  // Reference implementation
  int out_cpu_s = reference_blas::iamax(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<tuple_t>(int(1));
  ex.get_policy_handler().copy_to_device(&out_s, gpu_out_s, 1);

  _iamax(ex, size, gpu_x_v, incX, gpu_out_s);
  auto event = ex.get_policy_handler().copy_to_host(gpu_out_s, &out_s, 1);
  ex.get_policy_handler().wait(event);

  using data_tuple_t = IndexValueTuple<int, data_t>;
  data_tuple_t out_data_s{out_s.ind, static_cast<data_t>(out_s.val)};

  // Validate the result
  ASSERT_EQ(out_cpu_s, out_data_s.ind);
  ASSERT_EQ(x_v[out_data_s.ind * incX], out_data_s.val);
  ASSERT_EQ(x_v[out_cpu_s * incX], out_data_s.val);
}

BLAS_REGISTER_TEST(Iamax, combination_t, combi);
