#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <interface/blas1_interface_sycl.hpp>

using namespace cl::sycl;
using namespace blas;

#define DEF_SIZE_VECT 1200
#define ERROR_ALLOWED 1.0E-6
// #define SHOW_VALUES   1

int main(int argc, char *argv[]) {
  size_t sizeV, returnVal = 0;
  double res;

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
    std::vector<double> vX(sizeV), vY(sizeV), vZ(sizeV), vR(1), vS(1), vT(1),
        vU(1);
    std::vector<IndVal<double>> vImax(
        1, IndVal<double>(std::numeric_limits<size_t>::max(),
                          std::numeric_limits<double>::min())),
        vImin(1, IndVal<double>(std::numeric_limits<size_t>::max(),
                                std::numeric_limits<double>::max()));

    size_t vSeed, gap;
    double minV, maxV;

    // INITIALIZING DATA
    vSeed = time(NULL) / 10 * 10;

    minV = -10.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vX), std::end(vX),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });

    minV = -30.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    std::for_each(std::begin(vY), std::end(vY),
                  [&](double &elem) { elem = minV + (double)(rand() % gap); });

    // COMPUTING THE RESULTS
    size_t i = 0, indMax = 0, indMin = 0;
    double sum = 0.0, alpha = 1.1, dot = 0.0;
    double nrmX = 0.0, nrmY = 0.0, max = 0.0, min = 1e9;
    double diff = 0.0;
    double _cos, _sin, giv = 0.0;
    std::for_each(std::begin(vZ), std::end(vZ), [&](double &elem) {
      elem = vY[i] + alpha * vX[i];
      sum += std::abs(elem);
      dot += (elem * vX[i]);
      nrmX += vX[i] * vX[i];
      nrmY += elem * elem;
      if (std::abs(elem) > std::abs(max)) {
        max = elem, indMax = i;
      }
      if (std::abs(elem) < std::abs(min)) {
        min = elem, indMin = i;
      }
      if (i == 0) {
        diff = elem - vX[i];
        double num1 = vX[0], num2 = elem;
        blas::_rotg(num1, num2, _cos, _sin);
      }
      giv += ((vX[i] * _cos + elem * _sin) * (elem * _cos - vX[i] * _sin));
      if (i == 0) {
        diff = (elem * _cos - vX[i] * _sin) - ((vX[i] * _cos + elem * _sin));
      } else if ((i + 1) == sizeV) {
        diff += (elem * _cos - vX[i] * _sin) - ((vX[i] * _cos + elem * _sin));
      }
      i++;
    });
    nrmX = std::sqrt(nrmX);
    nrmY = std::sqrt(nrmY);

    // CREATING THE SYCL QUEUE AND EXECUTOR
    cl::sycl::queue q([=](cl::sycl::exception_list eL) {
      try {
        for (auto &e : eL) {
          std::rethrow_exception(e);
        }
      } catch (cl::sycl::exception &e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });
    Executor<SYCL> ex(q);
    {
      // CREATION OF THE BUFFERS
      buffer<double, 1> bX(vX.data(), range<1>{vX.size()}),
          bY(vY.data(), range<1>{vY.size()}),
          bZ(vZ.data(), range<1>{vZ.size()}),
          bR(vR.data(), range<1>{vR.size()}),
          bS(vS.data(), range<1>{vS.size()}),
          bT(vT.data(), range<1>{vT.size()}),
          bU(vU.data(), range<1>{vU.size()});
      buffer<IndVal<double>, 1> bImax(vImax.data(), range<1>{vImax.size()}),
          bImin(vImin.data(), range<1>{vImin.size()});
      // BUILDING A SYCL VIEW OF THE BUFFERS
      BufferVectorView<double> bvX(bX), bvY(bY), bvZ(bZ), bvR(bR), bvS(bS),
          bvT(bT), bvU(bU);
      BufferVectorView<IndVal<double>> bvImax(bImax), bvImin(bImin);

      // EXECUTION OF THE ROUTINES
      _axpy<SYCL>(ex, bX.get_count(), alpha, bvX, 1, bvY, 1);
      _asum<SYCL>(ex, bY.get_count(), bvY, 1, bvR);
      /* vS[0] = _dot<SYCL>(ex, bY.get_count(), bvX, 1, bvY, 1); */
      _dot<SYCL>(ex, bY.get_count(), bvX, 1, bvY, 1, bvS);
      _nrm2<SYCL>(ex, bY.get_count(), bvY, 1, bvT);
      _iamax<SYCL>(ex, bY.get_count(), bvY, 1, bvImax);
      _iamin<SYCL>(ex, bY.get_count(), bvY, 1, bvImin);
      _rot<SYCL>(ex, bY.get_count(), bvX, 1, bvY, 1, _cos, _sin);
      _dot<SYCL>(ex, bY.get_count(), bvX, 1, bvY, 1, bvU);
      _swap<SYCL>(ex, bY.get_count(), bvX, 1, bvY, 1);
    }

    // ANALYSIS OF THE RESULTS
    res = vR[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , sum = " << sum
              << " , err = " << sum - res << std::endl;
#endif  //  SHOW_VALUES
    if (std::abs((res - sum) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , sum = " << sum
                << " , err = " << sum - res << std::endl;
      returnVal += 2;
    }

    res = vS[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , dot = " << dot
              << " , err = " << dot - res << std::endl;
#endif  //  SHOW_VALUES
    if (std::abs((res - dot) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , dot = " << dot
                << " , err = " << dot - res << std::endl;
      returnVal += 4;
    }

    res = vT[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , nrmY = " << nrmY
              << " , err = " << nrmY - res << std::endl;
#endif  //  SHOW_VALUES
    if (std::abs((res - nrmY) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , nrmY = " << nrmY
                << " , err = " << nrmY - res << std::endl;
      returnVal += 8;
    }

    IndVal<double> ind = vImax[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> resInd = " << ind.getInd()
              << ", resMax = " << ind.getVal() << " , ind = " << indMax
              << " , max = " << max << std::endl;
#endif  //  SHOW_VALUES
    if (ind.getInd() != indMax) {
      std::cout << "ERROR!! --> resInd = " << ind.getInd()
                << ", resMax = " << ind.getVal() << " , ind = " << indMax
                << " , max = " << max << std::endl;
      returnVal += 16;
    }

    ind = vImin[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> resInd = " << ind.getInd()
              << ", resmin = " << ind.getVal() << " , ind = " << indMin
              << " , min = " << min << std::endl;
#endif  //  SHOW_VALUES
    if (ind.getInd() != indMin) {
      std::cout << "ERROR!! --> resInd = " << ind.getInd()
                << ", resmin = " << ind.getVal() << " , ind = " << indMin
                << " , min = " << min << std::endl;
      returnVal += 16;
    }

    res = vU[0];
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , giv = " << giv
              << " , err = " << giv - res << std::endl;
#endif  //  SHOW_VALUES
    if (std::abs((res - giv) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , giv = " << giv
                << " , err = " << giv - res << std::endl;
      returnVal += 32;
    }

    res = (vX[0] - vY[0]) + (vX[sizeV - 1] - vY[sizeV - 1]);
#ifdef SHOW_VALUES
    std::cout << "VALUES!! --> res = " << res << " , diff = " << diff
              << " , err = " << diff - res << std::endl;
#endif  //  SHOW_VALUES
    if (std::abs((res - diff) / res) > ERROR_ALLOWED) {
      std::cout << "ERROR!! --> res = " << res << " , diff = " << diff
                << " , err = " << diff - res << std::endl;
      returnVal += 64;
    }
  }

  return returnVal;
}
