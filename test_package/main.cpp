#include <sycl-blas/interface/blas1_interface.hpp>
#include <memory>

using namespace blas;

template <typename ScalarT>
ScalarT *new_data(size_t size, bool initialized = true) {
  ScalarT *v = new ScalarT[size];
  if (initialized) {
    for (size_t i = 0; i < size; ++i) {
      v[i] = 1e-3 * ((rand() % 2000) - 1000);
    }
  }
  return v;
}

int main() {
    cl::sycl::queue q;
    Executor<SYCL> ex{q};

    const size_t size = 128;
    auto v1 = std::unique_ptr<float>(new_data<float>(size));
    auto v2 = std::unique_ptr<float>(new_data<float>(size));
    float alpha(2.4367453465);
    double flops;
    auto inx = ex.template allocate<float>(size);
    auto iny = ex.template allocate<float>(size);
    ex.copy_to_device(v1.get(), inx, size);
    ex.copy_to_device(v2.get(), iny, size);

    _axpy(ex, size, alpha, inx, 1, iny, 1);
    ex.sycl_queue().wait_and_throw();

    ex.template deallocate<float>(inx);
    ex.template deallocate<float>(iny);
}