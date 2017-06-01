#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <interface/blas2_interface_sycl.hpp>
#include <interface/blas3_interface_sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cl::sycl;
using namespace blas;

// #define VERBOSE  1

#define DEF_SIZE_VECT 1200
#define ERROR_ALLOWED 1.0E-8
#define RANDOM_DATA 1
#define EXECUTED_ON_GPU 1
// #define SHOW_VALUES   1

#ifdef EXECUTED_ON_GPU
#define DEFAULT_ACCESS false
#else
#define DEFAULT_ACCESS true
#endif

// INITIAL MATRIZ VECTOR PRODUCT

/*! TestingGEMM.
 * @brief Tests that GEMM works properly.
 *
 * @param bool accessDev
 * @param size_t dim
 * @param size_t divSz
 * @param size_t shftR
 * @param size_t shftC
 */
size_t TestingGEMM(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                   size_t shftC) {
  // CREATING DATA
  size_t dimR = dim / divSz;
  size_t dimC = dim * divSz;
  size_t dimM = dimR;
  size_t dimK = dimC + dimR;
  size_t dimN = dimC;
  size_t shftK = shftR + shftC;
  std::vector<double> vA(dimM * dimK);
  std::vector<double> vB(dimK * dimN);
  std::vector<double> vC(dimM * dimN);
  std::vector<double> vS(dimM * dimN);
  std::vector<double> vR(1);

  // INITIALIZING DATA
  size_t vSeed, gap;
  double minV, maxV;
#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vA), std::end(vA), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vB), std::end(vB), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  //  RANDOM_DATA

  std::for_each(std::begin(vC), std::end(vC), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

  // CREATING HOST STRUCTURES
  size_t dimLA = ((accessDev) ? dimK : dimM);
  matrix_view<double, std::vector<double>> m_A(vA, accessDev, dimM, dimK, true,
                                               dimLA, 0);
  size_t dimLB = ((accessDev) ? dimN : dimK);
  matrix_view<double, std::vector<double>> m_B(vB, accessDev, dimK, dimN, true,
                                               dimLB, 0);
  size_t dimLC = ((accessDev) ? dimN : dimM);
  matrix_view<double, std::vector<double>> m_C(vC, accessDev, dimM, dimN, true,
                                               dimLC, 0);
  matrix_view<double, std::vector<double>> m_S(vS, accessDev, dimM, dimN, true,
                                               dimLC, 0);
#ifdef VERBOSE
  m_A.printH("MA");
  m_B.printH("MB");
  m_C.printH("MC");
#endif

  // COMPUTING THE RESULTS

  size_t returnVal = 0;
  double res, alpha = 1.5, beta = 2.5;
#ifdef BLAS_EXPERIMENTAL
  v_A.printH("VA");
  v_B.printH("VB");
  v_C.printH("VC");
  // C = A * B
  for (size_t i = shftR; i < dimM; i++) {
    for (size_t j = shftR; j < dimN; j++) {
      if (accessDev)
        vS[dimN * i + j] = beta * vC[dimN * i + j];
      else
        vS[dimM * j + i] = beta * vC[dimM * j + i];
      for (size_t k = shftC; k < dimK; k++) {
        std::cout << "(" << i << "," << k << ")=" << dimM * k + i << std::endl;
        std::cout << "(" << k << "," << j << ")=" << dimK * j + k << std::endl;
        if (accessDev) {
          vS[dimN * i + j] += alpha * vA[dimK * i + k] * vB[dimN * k + j];
        } else {
          vS[dimM * j + i] += alpha * vA[dimM * k + i] * vB[dimK * j + k];
        }
      }
    }
  }
  v_S.printH("VS");
#endif  // BLAS_EXPERIMENTAL

  double addC = 0.0;
  // C= At * B
  for (size_t i = shftR; i < dimM; i++) {
    for (size_t j = shftR; j < dimN; j++) {
      if (accessDev)
        vS[dimN * i + j] = beta * vC[dimN * i + j];
      else
        vS[dimM * j + i] = beta * vC[dimM * j + i];
      for (size_t k = shftC; k < dimK; k++) {
        // c[i,j] += a[k,i] * b[k,j]
        if (accessDev) {
          vS[dimN * i + j] += alpha * vA[dimM * k + i] * vB[dimN * k + j];
        } else {
          vS[dimM * j + i] += alpha * vA[dimM * k + i] * vB[dimN * k + j];
        }
      }
      addC += (accessDev) ? vS[dimN * i + j] : vS[dimM * j + i];
    }
  }
#ifdef VERBOSE
  m_S.printH("MS");
#endif  // VERBOSE
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
    buffer<double, 1> bA0(vA.data(), range<1>{vA.size()});
    buffer<double, 1> bB0(vB.data(), range<1>{vB.size()});
    buffer<double, 1> bC0(vC.data(), range<1>{vC.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});

    // BUILDING A SYCL VIEW OF THE BUFFERS
    BufferMatrixView<double> bmA0(bA0, accessDev, dimM, dimK);
    BufferMatrixView<double> bmB0(bB0, accessDev, dimK, dimN);
    BufferMatrixView<double> bmC0(bC0, accessDev, dimM, dimN);
    BufferVectorView<double> bvV0(bC0);
    BufferVectorView<double> bvR(bR);

    // EXECUTION OF THE ROUTINES
    // size_t dimLA = ((accessDev) ? dimK : dimM); // Original
    size_t dimLA = dimM;  // for accessDev = true  then A^t
    // size_t dimLB = ((accessDev) ? dimN : dimK);  // Original
    size_t dimLB = dimN;  // for accessDev = false then B^t
    _gemm<SYCL>(ex, ((accessDev) ? "Tr" : "No"), ((accessDev) ? "No" : "Tr"),
                dimR - shftR, dimC - shftC, dimK - shftK, 1.5,
                bmA0(shftR, shftK), dimLA, bmB0(shftK, shftC), dimLB, 2.5,
                bmC0(shftR, shftC), dimLC);

    auto reducOpV = make_addAssignReduction(bvR, bvV0, 256, 512 * 256);
    ex.reduce(reducOpV);
  }
#ifdef VERBOSE
  m_C.printH("MC");
#endif  // VERBOSE

  // ANALYSIS OF THE RESULTS
  res = vR[0];
#ifdef SHOW_VALUES
  std::cout << "VALUES!! --> res = " << res << " , addC = " << addC
            << " , err = " << addC - res << std::endl;
#endif  //  SHOW_VALUES
  if (std::abs((res - addC) / res) > ERROR_ALLOWED) {
    std::cout << "ERROR!! --> res = " << res << " , addC = " << addC
              << " , err = " << addC - res << std::endl;
    returnVal += 1;
  }

  return returnVal;
}

int main(int argc, char *argv[]) {
  //  using namespace SyclBlas;
  //  bool accessDev = true;
  bool accessDev = DEFAULT_ACCESS;
  size_t sizeV = 0, divSz = 1, shftR = 0, shftC = 0;
  size_t returnVal = 0;

  if (argc == 1) {
    sizeV = DEF_SIZE_VECT;
  } else if (argc == 2) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
  } else if (argc == 3) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    divSz = atoi(argv[2]);
    ;
  } else if (argc == 4) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    shftR = atoi(argv[2]);
    shftC = atoi(argv[3]);
  } else if (argc == 5) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      //      accessDev = false;
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
    divSz = atoi(argv[2]);

    shftR = atoi(argv[3]);
    shftC = atoi(argv[4]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }

  if (returnVal == 0)
    returnVal = 2 * TestingGEMM(accessDev, sizeV, divSz, shftR, shftC);

  return returnVal;
}
