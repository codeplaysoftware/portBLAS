#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <interface/blas2_interface_sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cl::sycl;
using namespace blas;

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

template <typename ExecutorType, typename T, typename ContainerT>
void _gemv0(Executor<ExecutorType> ex, std::string _Trans, size_t _M, size_t _N,
            T _alpha, matrix_view<T, ContainerT> _mA, size_t _lda,
            vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
            vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 'N') && (_Trans[0] != 't') &&
      (_Trans[0] != 'T') && (_Trans[0] != 'c') && (_Trans[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  bool accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  size_t M = (accessOpr ? _M : _N);
  size_t N = (accessOpr ? _N : _M);
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, M);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {
    auto scalOp = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto assignOp = make_op<Assign>(my_vy, scalOp);
    ex.execute(assignOp);
    auto disp = my_mA.getDisp();
    for (size_t i = 0; i < M; i++) {
      auto my_row = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, N);
      auto my_rs = vector_view<T, ContainerT>(my_vy, i + my_vy.getDisp(), 1, 1);
      auto scl = my_rs.eval(0);
      auto prdOp1 = make_op<BinaryOp, prdOp2_struct>(my_row, my_vx);
      auto localSize = 256;
      auto nWG = 128;
#ifdef SYCL_CODE
      auto assignOp1 =
          make_addAssignReduction(my_rs, prdOp1, localSize, localSize * nWG);
      ex.reduce(assignOp1);
#else   // SYCL_CODE
      ContainerT valT1(nWG);
      auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
      auto assignOp11 =
          make_addAssignReduction(val1, prdOp1, localSize, localSize * nWG);
      ex.execute(assignOp11);
      auto assignOp1 = make_addAssignReduction(my_rs, val1, localSize, nWG);
      ex.execute(assignOp1);
#endif  // SYCL_CODE
      auto prdOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, my_rs);
      auto addOp2 = make_op<ScalarOp, addOp2_struct>(scl, prdOp2);
      auto assignOp2 = make_op<Assign>(my_rs, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  } else {
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto assignOp1 = make_op<Assign>(my_vy, scalOp1);
    ex.execute(assignOp1);
    auto disp = my_mA.getDisp();
    for (size_t j = 0; j < N; j++) {
      auto my_col = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, M);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vx.eval(j), my_col);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_vy, scalOp2);
      auto assignOp2 = make_op<Assign>(my_vy, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif  // VERBOSE
}

// INITIAL RANK 1 MODIFICATION

template <typename ExecutorType, typename T, typename ContainerT>
void _ger0(Executor<ExecutorType> ex, size_t _M, size_t _N, T _alpha,
           vector_view<T, ContainerT> _vx, size_t _incx,
           vector_view<T, ContainerT> _vy, size_t _incy,
           matrix_view<T, ContainerT> _mA, size_t _lda) {
  bool accessOpr = true;
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, M);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {
    auto disp = my_mA.getDisp();
    for (size_t i = 0; i < M; i++) {
      auto my_row = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, N);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vx.eval(i), my_vy);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_row, scalOp2);
      auto assignOp2 = make_assign(my_row, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  } else {
    auto disp = my_mA.getDisp();
    for (size_t j = 0; j < N; j++) {
      auto my_col = vector_view<T, ContainerT>(my_mA.getData(), disp, 1, M);
      auto scalOp2 =
          make_op<ScalarOp, prdOp2_struct>(_alpha * my_vy.eval(j), my_vx);
      auto addOp2 = make_op<BinaryOp, addOp2_struct>(my_col, scalOp2);
      auto assignOp2 = make_assign(my_col, addOp2);
      ex.execute(assignOp2);
      disp += _lda;
    }
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif  // VERBOSE
}

// TESTING ROUTINE

size_t TestingBLAS2(bool accessDev, size_t dim, size_t divSz, size_t shftR,
                    size_t shftC) {
  // CREATING DATA
  size_t dimR = dim / divSz;
  size_t dimC = dim * divSz;
  std::vector<double> vM(dimR * dimC);
  std::vector<double> vX(dimC);
  std::vector<double> vY(dimR);
  std::vector<double> vX2(dimC);
  std::vector<double> vY2(dimR);
  std::vector<double> vR(1);
  std::vector<double> vS(1);
  std::vector<double> vT(1);

  // INITIALIZING DATA
  size_t vSeed, gap;
  double minV, maxV;
#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vM), std::end(vM), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA

  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vX), std::end(vX), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 1.00;
  maxV = 1.00;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vY), std::end(vY), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

  // CREATING HOST STRUCTURES
  size_t dimL = ((accessDev) ? dimC : dimR);
  matrix_view<double, std::vector<double>> v_M0(vM, accessDev, dimR, dimC, true,
                                                dimL, 0);
  vector_view<double, std::vector<double>> v_X0(vX, 0, 1, dimC);
  vector_view<double, std::vector<double>> v_Y0(vY, 0, 1, dimR);
  vector_view<double, std::vector<double>> v_R(vR, 0, 1, 1);

  // COMPUTING THE RESULTS
  size_t returnVal = 0;
  double res;

  double addY = 0.0;
  for (size_t i = shftR; i < dimR; i++) {
    vY2[i - shftR] = 1.5 * vY[i - shftR];
    for (size_t j = shftC; j < dimC; j++) {
      if (accessDev) {
        vY2[i - shftR] += 2.0 * vM[dimC * i + j] * vX[j - shftC];
      } else {
        vY2[i - shftR] += 2.0 * vM[dimR * j + i] * vX[j - shftC];
      }
    }
    addY += vY2[i - shftR];
  }
  for (size_t i = dimR - shftR; i < dimR; i++) {
#ifdef VERBOSE
    std::cout << "+" << vY[i] << std::endl;
#endif  // VERBOSE
    addY += vY[i];
  }

  double addX = 0.0;
  for (size_t j = shftC; j < dimC; j++) {
    vX2[j - shftC] = 0.5 * vX[j - shftC];
    for (size_t i = shftR; i < dimR; i++) {
      if (accessDev) {
        vX2[j - shftC] += 2.5 * vM[dimC * i + j] * vY2[i - shftR];
      } else {
        vX2[j - shftC] += 2.5 * vM[dimR * j + i] * vY2[i - shftR];
      }
    }
    addX += vX2[j - shftC];
  }
  for (size_t j = dimC - shftC; j < dimC; j++) {
    addX += vX[j];
  }

  double addRng1 = 0.0;
  for (size_t i = 0; i < dimR; i++) {
    for (size_t j = 0; j < dimC; j++) {
      addRng1 += (accessDev) ? vM[dimC * i + j] : vM[dimR * j + i];
      if ((i >= shftR) && (j >= shftC)) {
        addRng1 += 3.0 * vY2[i - shftR] * vX2[j - shftC];
      }
    }
  }

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
    buffer<double, 1> bM0(vM.data(), range<1>{vM.size()});
    buffer<double, 1> bX0(vX.data(), range<1>{vX.size()});
    buffer<double, 1> bY0(vY.data(), range<1>{vY.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<double, 1> bS(vS.data(), range<1>{vS.size()});
    buffer<double, 1> bT(vT.data(), range<1>{vT.size()});

    // BUILDING A SYCL VIEW OF THE BUFFERS
    BufferMatrixView<double> bmM0(bM0, accessDev, dimR, dimC);
    BufferVectorView<double> bvV0(bM0);
    BufferVectorView<double> bvX0(bX0);
    BufferVectorView<double> bvY0(bY0);
    BufferVectorView<double> bvR(bR);
    BufferVectorView<double> bvS(bS);
    BufferVectorView<double> bvT(bT);

    // EXECUTION OF THE ROUTINES
    _gemv<SYCL>(ex, "No", dimR - shftR, dimC - shftC, 2.0, bmM0(shftR, shftC),
                dimL, bvX0, 1, 1.5, bvY0, 1);
    _gemv<SYCL>(ex, "Tr", dimC - shftC, dimR - shftR, 2.5, bmM0(shftR, shftC),
                dimL, bvY0, 1, 0.5, bvX0, 1);

    _ger<SYCL>(ex, dimR - shftR, dimC - shftC, 3.0, bvY0, 1, bvX0, 1,
               bmM0(shftR, shftC), dimL);
    auto reducOpX = make_addAssignReduction(bvR, bvX0, 256, 512 * 256);
    ex.reduce(reducOpX);
    auto reducOpY = make_addAssignReduction(bvS, bvY0, 256, 512 * 256);
    ex.reduce(reducOpY);
    auto reducOpV = make_addAssignReduction(bvT, bvV0, 256, 512 * 256);
    ex.reduce(reducOpV);
  }

  // ANALYSIS OF THE RESULTS
  res = vR[0];
#ifdef SHOW_VALUES
  std::cout << "VALUES!! --> res = " << res << " , addX = " << addX
            << " , err = " << addX - res << std::endl;
#endif  // VERBOSE
  if (std::abs((res - addX) / res) > ERROR_ALLOWED) {
    std::cout << "ERROR!! --> res = " << res << " , addX = " << addX
              << " , err = " << addX - res << std::endl;
    returnVal += 1;
  }

  res = vS[0];
#ifdef SHOW_VALUES
  std::cout << "VALUES!! --> res = " << res << " , addY = " << addY
            << " , err = " << addY - res << std::endl;
#endif  // VERBOSE
  if (std::abs((res - addY) / res) > ERROR_ALLOWED) {
    std::cout << "ERROR!! --> res = " << res << " , addY = " << addY
              << " , err = " << addY - res << std::endl;
    returnVal += 2;
  }

  res = vT[0];
#ifdef SHOW_VALUES
  std::cout << "VALUES!! --> res = " << res << " , addRng1 = " << addRng1
            << " , err = " << addRng1 - res << std::endl;
#endif  // VERBOSE
  if (std::abs((res - addRng1) / res) > ERROR_ALLOWED) {
    std::cout << "ERROR!! --> res = " << res << " , addRng1 = " << addRng1
              << " , err = " << addRng1 - res << std::endl;
    returnVal += 2;
  }

  return returnVal;
}

int main(int argc, char *argv[]) {
  //  using namespace SyclBlas;
  bool accessDev = DEFAULT_ACCESS;
  size_t sizeV = 0, divSz = 1, shftR = 0, shftC = 0;
  size_t returnVal = 0;

  if (argc == 1) {
    sizeV = DEF_SIZE_VECT;
  } else if (argc == 2) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
      accessDev = !DEFAULT_ACCESS;
    } else
      sizeV = atoi(argv[1]);
  } else if (argc == 3) {
    if (atoi(argv[1]) < 0) {
      sizeV = -atoi(argv[1]);
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
    ;
    shftR = atoi(argv[3]);
    shftC = atoi(argv[4]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    returnVal = 1;
  }

  if (returnVal == 0)
    returnVal = 2 * TestingBLAS2(accessDev, sizeV, divSz, shftR, shftC);

  return returnVal;
}
