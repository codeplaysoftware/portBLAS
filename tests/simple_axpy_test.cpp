#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cl::sycl;
using namespace blas;

#define DEF_SIZE_VECT 1200
#define ERROR_ALLOWED 1.0E-6
// #define SHOW_VALUES   1

int main(int argc, char *argv[]) {
  size_t sizeV, returnVal = 0;

  if (argc == 1) {
    sizeV = DEF_SIZE_VECT;
  } else if (argc == 2) {
    sizeV = atoi(argv[1]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }
  if (returnVal == 0) {
    // CREATING DATA
    std::vector<double> vX(sizeV);
    std::vector<double> vY(sizeV);
    double sum = 0.0f;

    // INITIALIZING DATA
    std::for_each(std::begin(vX), std::end(vX),
                  [&](double &elem) { elem = 1.0; });

    std::for_each(std::begin(vY), std::end(vY),
                  [&](double &elem) { elem = 1.0; });

    // CREATING THE SYCL QUEUE AND EXECUTOR
    cl::sycl::queue q([=](cl::sycl::exception_list eL) {
      for (auto &e : eL) {
        try {
            std::rethrow_exception(e);
        } catch (cl::sycl::exception &e) {
          std::cout << " E " << e.what() << std::endl;
        } catch (...) {
          std::cout << " An exception " << std::endl;
        }
      }
    });
    SYCLDevice dev(q);

    {
      // CREATION OF THE BUFFERS
      buffer<double, 1> bX(vX.data(), range<1>{vX.size()});
      buffer<double, 1> bY(vY.data(), range<1>{vY.size()});

      // BUILDING A SYCL VIEW OF THE BUFFERS
      BufferVectorView<double> bvX(bX);
      BufferVectorView<double> bvY(bY);

      // EXECUTION OF THE ROUTINES
      blas::execute(dev, _axpy(DEF_SIZE_VECT, 1.0, bvX, 0, 1, bvY, 0, 1));
    }

    std::cout << " Output: ";
    std::for_each(std::begin(vX), std::end(vX), [&](double elem) {
      std::cout << elem << ",";
    });
    std::cout << std::endl;
    std::for_each(std::begin(vY), std::end(vY), [&](double elem) {
      std::cout << elem << ",";
      sum += elem;
    });
    std::cout << std::endl;
    returnVal = (sum == (sizeV * (sizeV + 1) / 2.0));
  }

  return returnVal;
}
