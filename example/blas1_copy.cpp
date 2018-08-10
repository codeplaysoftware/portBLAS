#include <iostream>

#include "queue/sycl_iterator.hpp"
#include <interface/blas1_interface.hpp>
#include <interface/blas2_interface.hpp>
#include <interface/blas3_interface.hpp>

using namespace blas;

int main(int argc, char const *argv[]) {
  using ScalarT = float;
  std::cout << "Example building, cmake working" << std::endl;

  size_t size = 128;
  long strd = 8;

  std::cout << "size == " << size << std::endl;
  std::cout << "strd == " << strd << std::endl;

  // create two vectors: vX and vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size, 0);

  // initialise the vector
  ScalarT left(-1), right(1);
  for (size_t i = 0; i < size; ++i) {
    vX[i] = rand() % int((right - left) * 8) * 0.125 - right;
  }

  cl::sycl::default_selector d;
  auto q = cl::sycl::queue(d, [=](cl::sycl::exception_list eL) {
    for (auto &e : eL) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << "E " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "Standard Exception " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    }
  });

  Executor<SYCL> ex(q);

  auto gpu_vX = blas::helper::make_sycl_iteator_buffer<ScalarT>(vX, size);
  auto gpu_vY = blas::helper::make_sycl_iteator_buffer<ScalarT>(size);

  try {
    _copy(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd);
  } catch (cl::sycl::exception &e) {
    std::cout << "E " << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cout << "Standard Exception " << e.what() << std::endl;
  } catch (...) {
    std::cout << " An exception " << std::endl;
  }

  ex.copy_to_host(gpu_vY, vY.data());
  // check that vX and vY are the same
  for (size_t i = 0; i < size; ++i) {
    if (i % strd == 0) {
      assert(vX[i] == vY[i]);
    } else {
      assert(0 == vY[i]);
    }
  }

  std::cout << "Finished execution" << std::endl;

  return 0;
}
