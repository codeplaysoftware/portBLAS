#include "blas_test.hpp"
#include "unittest/blas1/blas1_iaminmax_common.hpp"
#include <limits>

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using tuple_t = IndexValueTuple<int, scalar_t>;

  api_type api;
  index_t size;
  index_t incX;
  generation_mode_t mode;
  std::tie(api, size, incX, mode) = combi;

  // Input vector
  std::vector<scalar_t> x_v(size * incX);
  populate_data<scalar_t>(mode, 0.0, x_v);

  // This will remove infs from the vector
  std::transform(
      std::begin(x_v), std::end(x_v), std::begin(x_v),
      [](scalar_t v) { return utils::clamp_to_limits<scalar_t>(v); });

  // Output scalar
  tuple_t out_s{0, 0.0};

  // Reference implementation
  int out_cpu_s = reference_blas::iamax(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v =
      blas::helper::BlasUsmHelper<true, scalar_t>::allocate(size * incX, q);

  auto copy_x =
      blas::helper::copy_to_device(q, x_v.data(), gpu_x_v, size * incX);

  if (api == api_type::async) {
    auto gpu_out_s =
        blas::helper::BlasUsmHelper<true, scalar_t>::allocate(1, q);
    auto copy_out =
        blas::helper::copy_to_device<scalar_t>(q, &out_s, gpu_out_s, 1);
    sb_handle.wait(copy_out);
    _iamax(sb_handle, size, gpu_x_v, incX, gpu_out_s);
    auto event =
        blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s, &out_s, 1);
    sb_handle.wait(event);
  } else {
    out_s.ind = _iamax(sb_handle, size, gpu_x_v, incX);
  }

  // Validate the result
  ASSERT_EQ(out_cpu_s, out_s.ind);
}

BLAS_REGISTER_TEST_ALL(Iamax, combination_t, combi, generate_name);
