#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <operations/blas1_trees.hpp>
#include <interface/blas1_interface_sycl.hpp>

using namespace cl::sycl;
using namespace blas;

#define DEF_SIZE_VECT 1200
#define ERROR_ALLOWED 1.0E-6
// #define SHOW_VALUES   1

//#define SHOW_TIMES     1  // If it exists, the code prints the execution time
                          // The ... should be changed by the corresponding routine
#define NUMBER_REPEATS 2  // Number of times the computations are made
                          // If it is greater than 1, the compile time is not considered


// #########################
// #include <adds.hpp>
// #########################
template <typename ExecutorType, typename T, typename ContainerT>
void _one_add(Executor<ExecutorType> ex, int _N,
              vector_view<T, ContainerT> _vx, int _incx,
              vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
  make_addReducAssignNewOp2(my_rs, my_vx, localSize, localSize * nWG);
  ex.reduce(assignOp);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif
}

template <typename ExecutorType, typename T, typename ContainerT>
void _two_add(Executor<ExecutorType> ex, int _N,
              vector_view<T, ContainerT> _vx1, int _incx1,
              vector_view<T, ContainerT> _rs1,
              vector_view<T, ContainerT> _vx2, int _incx2,
              vector_view<T, ContainerT> _rs2) {
  // Common definitions
  auto localSize = 256;
  auto nWG = 512;

  // _rs1 = add(_vx1)
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_rs1 = vector_view<T, ContainerT>(_rs1, _rs1.getDisp(), 1, 1);
  auto assignOp1 =
      make_addReducAssignNewOp2(my_rs1, my_vx1, localSize, localSize * nWG);
  ex.reduce(assignOp1);

  // _rs2 = add(_vx2)
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_rs2 = vector_view<T, ContainerT>(_rs2, _rs2.getDisp(), 1, 1);
  auto assignOp2 =
  make_addReducAssignNewOp2(my_rs2, my_vx2, localSize, localSize * nWG);
  ex.reduce(assignOp2);
  
  // concatenate both operations
//  auto doubleAssignOp = make_op<Join>(assignOp1, assignOp2);

  // execute concatenated operations
//  ex.reduce(doubleAssignOp);
}

template <typename ExecutorType, typename T, typename ContainerT>
void _four_add(Executor<ExecutorType> ex, int _N,
               vector_view<T, ContainerT> _vx1, int _incx1,
               vector_view<T, ContainerT> _rs1,
               vector_view<T, ContainerT> _vx2, int _incx2,
               vector_view<T, ContainerT> _rs2,
               vector_view<T, ContainerT> _vx3, int _incx3,
               vector_view<T, ContainerT> _rs3,
               vector_view<T, ContainerT> _vx4, int _incx4,
               vector_view<T, ContainerT> _rs4) {
  // Common definitions
  auto localSize = 256;
  auto nWG = 512;
  
  // _rs1 = add(_vx1)
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_rs1 = vector_view<T, ContainerT>(_rs1, _rs1.getDisp(), 1, 1);
  auto assignOp1 =
  make_addReducAssignNewOp2(my_rs1, my_vx1, localSize, localSize * nWG);
  ex.reduce(assignOp1);
  
  // _rs2 = add(_vx2)
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_rs2 = vector_view<T, ContainerT>(_rs2, _rs2.getDisp(), 1, 1);
  auto assignOp2 =
  make_addReducAssignNewOp2(my_rs2, my_vx2, localSize, localSize * nWG);
  ex.reduce(assignOp2);
  
  // _rs3 = add(_vx3)
  auto my_vx3 = vector_view<T, ContainerT>(_vx3, _vx3.getDisp(), _incx3, _N);
  auto my_rs3 = vector_view<T, ContainerT>(_rs3, _rs3.getDisp(), 1, 1);
  auto assignOp3 =
  make_addReducAssignNewOp2(my_rs3, my_vx3, localSize, localSize * nWG);
  ex.reduce(assignOp3);
  
  // _rs4 = add(_vx4)
  auto my_vx4 = vector_view<T, ContainerT>(_vx4, _vx4.getDisp(), _incx4, _N);
  auto my_rs4 = vector_view<T, ContainerT>(_rs4, _rs4.getDisp(), 1, 1);
  auto assignOp4 =
  make_addReducAssignNewOp2(my_rs4, my_vx4, localSize, localSize * nWG);
  ex.reduce(assignOp4);

  // concatenate operations
//  auto doubleAssignOp12 = make_op<Join>(assignOp1, assignOp2);
//  auto doubleAssignOp34 = make_op<Join>(assignOp3, assignOp4);
//  auto quadAssignOp = make_op<Join>(doubleAssignOp12, doubleAssignOp34);
  
  // execute concatenated operations
//  ex.execute(quadAssignOp);
}

// #########################
// #include <axpys.hpp>
// #########################
template <typename ExecutorType, typename T, typename ContainerT>
void _one_axpy(Executor<ExecutorType> ex, int _N, T _alpha,
               vector_view<T, ContainerT> _vx, int _incx,
               vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, my_vx);
  auto addOp = make_op<BinaryOp, addOp2_struct>(my_vy, scalOp);
  auto assignOp = make_op<Assign>(my_vy, addOp);
  ex.execute(assignOp);
#ifdef VERBOSE
  my_vy.printH("VY");
#endif
}

template <typename ExecutorType, typename T, typename ContainerT>
void _two_axpy(Executor<ExecutorType> ex, int _N,
               double _alpha1,
               vector_view<T, ContainerT> _vx1, int _incx1,
               vector_view<T, ContainerT> _vy1, int _incy1,
               double _alpha2,
               vector_view<T, ContainerT> _vx2, int _incx2,
               vector_view<T, ContainerT> _vy2, int _incy2) {
  // _vy1 = _alpha1 * _vx1 + _vy1
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_vy1 = vector_view<T, ContainerT>(_vy1, _vy1.getDisp(), _incy1, _N);
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_alpha1, my_vx1);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(my_vy1, scalOp1);
  
  // _vy2 = _alpha2 * _vx2 + _vy2
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_vy2 = vector_view<T, ContainerT>(_vy2, _vy2.getDisp(), _incy2, _N);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha2, my_vx2);
  auto addBinaryOp2 = make_op<BinaryOp, addOp2_struct>(my_vy2, scalOp2);
  
  // concatenate both operations
  auto assignOp1 = make_op<Assign>(my_vy1, addBinaryOp1);
  auto assignOp2 = make_op<Assign>(my_vy2, addBinaryOp2);
  auto doubleAssignOp = make_op<Join>(assignOp1, assignOp2);
  
  // execute concatenated operations
  ex.execute(doubleAssignOp);
}

template <typename ExecutorType, typename T, typename ContainerT>
void _four_axpy(Executor<ExecutorType> ex, int _N,
                 double _alpha1,
                 vector_view<T, ContainerT> _vx1, int _incx1,
                 vector_view<T, ContainerT> _vy1, int _incy1,
                 double _alpha2,
                 vector_view<T, ContainerT> _vx2, int _incx2,
                 vector_view<T, ContainerT> _vy2, int _incy2,
                 double _alpha3,
                 vector_view<T, ContainerT> _vx3, int _incx3,
                 vector_view<T, ContainerT> _vy3, int _incy3,
                 double _alpha4,
                 vector_view<T, ContainerT> _vx4, int _incx4,
                 vector_view<T, ContainerT> _vy4, int _incy4) {
  // _vy1 = _alpha1 * _vx1 + _vy1
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_vy1 = vector_view<T, ContainerT>(_vy1, _vy1.getDisp(), _incy1, _N);
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_alpha1, my_vx1);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(my_vy1, scalOp1);
  
  // _vy2 = _alpha2 * _vx2 + _vy2
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_vy2 = vector_view<T, ContainerT>(_vy2, _vy2.getDisp(), _incy2, _N);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha2, my_vx2);
  auto addBinaryOp2 = make_op<BinaryOp, addOp2_struct>(my_vy2, scalOp2);
  
  // _vy3 = _alpha3 * _vx3 + _vy3
  auto my_vx3 = vector_view<T, ContainerT>(_vx3, _vx3.getDisp(), _incx3, _N);
  auto my_vy3 = vector_view<T, ContainerT>(_vy3, _vy3.getDisp(), _incy3, _N);
  auto scalOp3 = make_op<ScalarOp, prdOp2_struct>(_alpha3, my_vx3);
  auto addBinaryOp3 = make_op<BinaryOp, addOp2_struct>(my_vy3, scalOp3);
  
  // _vy4 = _alpha4 * _vx4 + _vy4
  auto my_vx4 = vector_view<T, ContainerT>(_vx4, _vx4.getDisp(), _incx4, _N);
  auto my_vy4 = vector_view<T, ContainerT>(_vy4, _vy4.getDisp(), _incy4, _N);
  auto scalOp4 = make_op<ScalarOp, prdOp2_struct>(_alpha4, my_vx4);
  auto addBinaryOp4 = make_op<BinaryOp, addOp2_struct>(my_vy4, scalOp4);
  
  // concatenate operations
  auto assignOp1 = make_op<Assign>(my_vy1, addBinaryOp1);
  auto assignOp2 = make_op<Assign>(my_vy2, addBinaryOp2);
  auto assignOp3 = make_op<Assign>(my_vy3, addBinaryOp3);
  auto assignOp4 = make_op<Assign>(my_vy4, addBinaryOp4);
  auto doubleAssignOp12 = make_op<Join>(assignOp1, assignOp2);
  auto doubleAssignOp34 = make_op<Join>(assignOp3, assignOp4);
  auto quadAssignOp = make_op<Join>(doubleAssignOp12, doubleAssignOp34);
  
  // execute concatenated operations
  ex.execute(quadAssignOp);
}


// #########################
// #include <copys.hpp>
// #########################
template <typename ExecutorType, typename T, typename ContainerT>
void _one_copy(Executor<ExecutorType> ex, int _N,
               vector_view<T, ContainerT> _vx, int _incx,
               vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif
  auto assignOp = make_op<Assign>(my_vy, my_vx);
  ex.execute(assignOp);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif
}

template <typename ExecutorType, typename T, typename ContainerT>
void _two_copy(Executor<ExecutorType> ex, int _N,
               vector_view<T, ContainerT> _vx1, int _incx1,
               vector_view<T, ContainerT> _vy1, int _incy1,
               vector_view<T, ContainerT> _vx2, int _incx2,
               vector_view<T, ContainerT> _vy2, int _incy2) {
  // _vy1 = _vx1
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_vy1 = vector_view<T, ContainerT>(_vy1, _vy1.getDisp(), _incy1, _N);
  auto assignOp1 = make_op<Assign>(my_vy1, my_vx1);
  
  // _vy2 = _vx2
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_vy2 = vector_view<T, ContainerT>(_vy2, _vy2.getDisp(), _incy2, _N);
  auto assignOp2 = make_op<Assign>(my_vy2, my_vx2);
  
  // concatenate both operations
  auto doubleAssignOp = make_op<Join>(assignOp1, assignOp2);
  
  // execute concatenated operations
  ex.execute(doubleAssignOp);
}

template <typename ExecutorType, typename T, typename ContainerT>
void _four_copy(Executor<ExecutorType> ex, int _N,
               vector_view<T, ContainerT> _vx1, int _incx1,
               vector_view<T, ContainerT> _vy1, int _incy1,
               vector_view<T, ContainerT> _vx2, int _incx2,
               vector_view<T, ContainerT> _vy2, int _incy2,
               vector_view<T, ContainerT> _vx3, int _incx3,
               vector_view<T, ContainerT> _vy3, int _incy3,
               vector_view<T, ContainerT> _vx4, int _incx4,
               vector_view<T, ContainerT> _vy4, int _incy4) {
  // _vy1 = _vx1
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_vy1 = vector_view<T, ContainerT>(_vy1, _vy1.getDisp(), _incy1, _N);
  auto assignOp1 = make_op<Assign>(my_vy1, my_vx1);
  
  // _vy2 = _vx2
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_vy2 = vector_view<T, ContainerT>(_vy2, _vy2.getDisp(), _incy2, _N);
  auto assignOp2 = make_op<Assign>(my_vy2, my_vx2);
  
  // _vy3 = _vx3
  auto my_vx3 = vector_view<T, ContainerT>(_vx3, _vx3.getDisp(), _incx3, _N);
  auto my_vy3 = vector_view<T, ContainerT>(_vy3, _vy3.getDisp(), _incy3, _N);
  auto assignOp3 = make_op<Assign>(my_vy3, my_vx3);
  
  // _vy4 = _vx4
  auto my_vx4 = vector_view<T, ContainerT>(_vx4, _vx4.getDisp(), _incx4, _N);
  auto my_vy4 = vector_view<T, ContainerT>(_vy4, _vy4.getDisp(), _incy4, _N);
  auto assignOp4 = make_op<Assign>(my_vy4, my_vx4);
  
  // concatenate operations
  auto doubleAssignOp12 = make_op<Join>(assignOp1, assignOp2);
  auto doubleAssignOp34 = make_op<Join>(assignOp3, assignOp4);
  auto quadAssignOp = make_op<Join>(doubleAssignOp12, doubleAssignOp34);
  
  // execute concatenated operations
  ex.execute(quadAssignOp);
}

// #########################

int main(int argc, char* argv[]) {
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
#ifdef SHOW_TIMES
    // VARIABLES FOR TIMING
    double t_start, t_stop;
    double t0_copy, t0_axpy, t0_add;
    double t1_copy, t1_axpy, t1_add;
    double t2_copy, t2_axpy, t2_add;
    double t3_copy, t3_axpy, t3_add;
#endif

    // CREATING DATA
    std::vector<double> vX1(sizeV);
    std::vector<double> vY1(sizeV);
    std::vector<double> vZ1(sizeV);
    std::vector<double> vS1(4);
    std::vector<double> vX2(sizeV);
    std::vector<double> vY2(sizeV);
    std::vector<double> vZ2(sizeV);
    std::vector<double> vS2(4);
    std::vector<double> vX3(sizeV);
    std::vector<double> vY3(sizeV);
    std::vector<double> vZ3(sizeV);
    std::vector<double> vS3(4);
    std::vector<double> vX4(sizeV);
    std::vector<double> vY4(sizeV);
    std::vector<double> vZ4(sizeV);
    std::vector<double> vS4(4);

    // INITIALIZING DATA
    size_t vSeed, gap;
    double minV, maxV;

    vSeed = 1;
    minV = -10.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vX1), std::end(vX1),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX2), std::end(vX2),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX3), std::end(vX3),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vX4), std::end(vX4),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });

    vSeed = 1;
    minV = -30.0;
    maxV = 10.0;
    gap = (size_t)(maxV - minV + 1);
    srand(vSeed);
    std::for_each(std::begin(vZ1), std::end(vZ1),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ2), std::end(vZ2),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ3), std::end(vZ3),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });
    std::for_each(std::begin(vZ4), std::end(vZ4),
                  [&](double& elem) { elem = minV + (double)(rand() % gap); });

    // COMPUTING THE RESULTS
    int i;
    double sum1 = 0.0f, alpha1 = 1.1f;
    double sum2 = 0.0f, alpha2 = 2.2f;
    double sum3 = 0.0f, alpha3 = 3.3f;
    double sum4 = 0.0f, alpha4 = 4.4f;
    double ONE = 1.0f;
    i = 0;
    std::for_each(std::begin(vY1), std::end(vY1), [&](double& elem) {
      elem = vZ1[i] + alpha1 * vX1[i]; sum1 += elem; i++;
    });
//    vS1[0] = sum1;
    i = 0;
    std::for_each(std::begin(vY2), std::end(vY2), [&](double& elem) {
      elem = vZ2[i] + alpha2 * vX2[i]; sum2 += elem; i++;
    });
//    vS2[0] = sum2;
    i = 0;
    std::for_each(std::begin(vY3), std::end(vY3), [&](double& elem) {
      elem = vZ3[i] + alpha3 * vX3[i]; sum3 += elem; i++;
    });
//    vS3[0] = sum3;
    i = 0;
    std::for_each(std::begin(vY4), std::end(vY4), [&](double& elem) {
      elem = vZ4[i] + alpha4 * vX4[i]; sum4 += elem; i++;
    });
//    vS4[0] = sum4;

    // CREATING THE SYCL QUEUE AND EXECUTOR
    cl::sycl::queue q([=](cl::sycl::exception_list eL) {
      try {
        for (auto& e : eL) {
          std::rethrow_exception(e);
        }
      } catch (cl::sycl::exception& e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });
    Executor<SYCL> ex(q);

    {
      // CREATION OF THE BUFFERS
      buffer<double, 1> bX1(vX1.data(), range<1>{vX1.size()});
      buffer<double, 1> bY1(vY1.data(), range<1>{vY1.size()});
      buffer<double, 1> bZ1(vZ1.data(), range<1>{vZ1.size()});
      buffer<double, 1> bS1(vS1.data(), range<1>{vS1.size()});
      buffer<double, 1> bX2(vX2.data(), range<1>{vX2.size()});
      buffer<double, 1> bY2(vY2.data(), range<1>{vY2.size()});
      buffer<double, 1> bZ2(vZ2.data(), range<1>{vZ2.size()});
      buffer<double, 1> bS2(vS2.data(), range<1>{vS2.size()});
      buffer<double, 1> bX3(vX3.data(), range<1>{vX3.size()});
      buffer<double, 1> bY3(vY3.data(), range<1>{vY3.size()});
      buffer<double, 1> bZ3(vZ3.data(), range<1>{vZ3.size()});
      buffer<double, 1> bS3(vS3.data(), range<1>{vS3.size()});
      buffer<double, 1> bX4(vX4.data(), range<1>{vX4.size()});
      buffer<double, 1> bY4(vY4.data(), range<1>{vY4.size()});
      buffer<double, 1> bZ4(vZ4.data(), range<1>{vZ4.size()});
      buffer<double, 1> bS4(vS4.data(), range<1>{vS4.size()});
      // BUILDING A SYCL VIEW OF THE BUFFERS
      BufferVectorView<double> bvX1(bX1);
      BufferVectorView<double> bvY1(bY1);
      BufferVectorView<double> bvZ1(bZ1);
      BufferVectorView<double> bvS1(bS1);
      BufferVectorView<double> bvX2(bX2);
      BufferVectorView<double> bvY2(bY2);
      BufferVectorView<double> bvZ2(bZ2);
      BufferVectorView<double> bvS2(bS2);
      BufferVectorView<double> bvX3(bX3);
      BufferVectorView<double> bvY3(bY3);
      BufferVectorView<double> bvZ3(bZ3);
      BufferVectorView<double> bvS3(bS3);
      BufferVectorView<double> bvX4(bX4);
      BufferVectorView<double> bvY4(bY4);
      BufferVectorView<double> bvZ4(bZ4);
      BufferVectorView<double> bvS4(bS4);

      for (int i=0; i<NUMBER_REPEATS; i++) {
        // EXECUTION OF THE ROUTINES (FOR CLBLAS)
#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_copy<SYCL>(ex, bX1.get_count(), bvZ1, 1, bvY1, 1);
        _one_copy<SYCL>(ex, bX2.get_count(), bvZ2, 1, bvY2, 1);
        _one_copy<SYCL>(ex, bX3.get_count(), bvZ3, 1, bvY3, 1);
        _one_copy<SYCL>(ex, bX4.get_count(), bvZ4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t0_copy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_axpy<SYCL>(ex, bX1.get_count(), alpha1, bvX1, 1, bvY1, 1);
        _one_axpy<SYCL>(ex, bX2.get_count(), alpha2, bvX2, 1, bvY2, 1);
        _one_axpy<SYCL>(ex, bX3.get_count(), alpha3, bvX3, 1, bvY3, 1);
        _one_axpy<SYCL>(ex, bX4.get_count(), alpha4, bvX4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t0_axpy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_add<SYCL>(ex, bY1.get_count(), bvY1, 1, bvS1);
        _one_add<SYCL>(ex, bY2.get_count(), bvY2, 1, bvS2);
        _one_add<SYCL>(ex, bY3.get_count(), bvY3, 1, bvS3);
        _one_add<SYCL>(ex, bY4.get_count(), bvY4, 1, bvS4);
#ifdef SHOW_TIMES
        t_stop  = ... ; t0_add  = t_stop - t_start ;
#endif

        // EXECUTION OF THE ROUTINES (SINGLE OPERATIONS)
#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_copy<SYCL>(ex, bX1.get_count(), bvZ1, 1, bvY1, 1);
        _one_copy<SYCL>(ex, bX2.get_count(), bvZ2, 1, bvY2, 1);
        _one_copy<SYCL>(ex, bX3.get_count(), bvZ3, 1, bvY3, 1);
        _one_copy<SYCL>(ex, bX4.get_count(), bvZ4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t1_copy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_axpy<SYCL>(ex, bX1.get_count(), alpha1, bvX1, 1, bvY1, 1);
        _one_axpy<SYCL>(ex, bX2.get_count(), alpha2, bvX2, 1, bvY2, 1);
        _one_axpy<SYCL>(ex, bX3.get_count(), alpha3, bvX3, 1, bvY3, 1);
        _one_axpy<SYCL>(ex, bX4.get_count(), alpha4, bvX4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t1_axpy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _one_add<SYCL>(ex, bY1.get_count(), bvY1, 1, bvS1+1);
        _one_add<SYCL>(ex, bY2.get_count(), bvY2, 1, bvS2+1);
        _one_add<SYCL>(ex, bY3.get_count(), bvY3, 1, bvS3+1);
        _one_add<SYCL>(ex, bY4.get_count(), bvY4, 1, bvS4+1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t1_add  = t_stop - t_start ;
#endif

        // EXECUTION OF THE ROUTINES (DOUBLE OPERATIONS)
#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _two_copy<SYCL>(ex, bX1.get_count(), bvZ1, 1, bvY1, 1,
                                             bvZ2, 1, bvY2, 1);
        _two_copy<SYCL>(ex, bX3.get_count(), bvZ3, 1, bvY3, 1,
                                             bvZ4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t2_copy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _two_axpy<SYCL>(ex, bX1.get_count(), alpha1, bvX1, 1, bvY1, 1,
                                             alpha2, bvX2, 1, bvY2, 1);
        _two_axpy<SYCL>(ex, bX3.get_count(), alpha3, bvX3, 1, bvY3, 1,
                                             alpha4, bvX4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t2_axpy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _two_add<SYCL>(ex, bY1.get_count(), bvY1, 1, bvS1+2,
                                            bvY2, 1, bvS2+2);
        _two_add<SYCL>(ex, bY3.get_count(), bvY3, 1, bvS3+2,
                                            bvY4, 1, bvS4+2);
#ifdef SHOW_TIMES
        t_stop  = ... ; t2_add  = t_stop - t_start ;
#endif

        // EXECUTION OF THE ROUTINES (QUADUBLE OPERATIONS)
#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _four_copy<SYCL>(ex, bX1.get_count(), bvZ1, 1, bvY1, 1,
                                             bvZ2, 1, bvY2, 1,
                                             bvZ3, 1, bvY3, 1,
                                             bvZ4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t3_copy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _four_axpy<SYCL>(ex, bX1.get_count(), alpha1, bvX1, 1, bvY1, 1,
                                             alpha2, bvX2, 1, bvY2, 1,
                                             alpha3, bvX3, 1, bvY3, 1,
                                             alpha4, bvX4, 1, bvY4, 1);
#ifdef SHOW_TIMES
        t_stop  = ... ; t3_axpy = t_stop - t_start ;
#endif

#ifdef SHOW_TIMES
        t_start = ... ; 
#endif
        _four_add<SYCL>(ex, bY1.get_count(), bvY1, 1, bvS1+3,
                                            bvY2, 1, bvS2+3,
                                            bvY3, 1, bvS3+3,
                                            bvY4, 1, bvS4+3);
#ifdef SHOW_TIMES
        t_stop  = ... ; t3_add  = t_stop - t_start ;
#endif
      }
    }
#ifdef SHOW_TIMES
    // COMPUTATIONAL TIMES
    std::cout <<   "t_copy --> (" << t0_copy << ", " << t1_copy 
                          << ", " << t2_copy << ", " << t3_copy << ")" << std::endl; 
    std::cout <<   "t_axpy --> (" << t0_axpy << ", " << t1_axpy 
                          << ", " << t2_axpy << ", " << t3_axpy << ")" << std::endl; 
    std::cout <<   "t_add  --> (" << t0_add  << ", " << t1_add  
                          << ", " << t2_add  << ", " << t3_add  << ")" << std::endl; 
#endif

    // ANALYSIS OF THE RESULTS
    double res;

    for (i=0; i<4; i++) {
      res = vS1[i]; 
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                << " , sum = " << sum1 << " , err = " << res - sum1 << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum1) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                  << " , sum = " << sum1 << " , err = " << res - sum1 << std::endl;
        returnVal += 2 * i;
      }
    }

    for (i=0; i<4; i++) {
      res = vS2[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                << " , sum = " << sum2 << " , err = " << res - sum2 << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs(( res - sum2) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                  << " , sum = " << sum2 << " , err = " << res - sum2 << std::endl;
        returnVal += 20 * i;
      }
    }


    for (i=0; i<4; i++) {
      res = vS3[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                << " , sum = " << sum3 << " , err = " << res - sum3 << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum3) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                  << " , sum = " << sum3 << " , err = " << res - sum3 << std::endl;
        returnVal += 200 * i;
      }
    }

    for (i=0; i<4; i++) {
      res = vS4[i];
#ifdef SHOW_VALUES
      std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                << " , sum = " << sum4 << " , err = " << res - sum4 << std::endl;
#endif  //  SHOW_VALUES
      if (std::abs((res - sum4) / res) > ERROR_ALLOWED) {
        std::cout << "VALUES!! --> res = " << res << " , i = " << i 
                  << " , sum = " << sum4 << " , err = " << res - sum4 << std::endl;
        returnVal += 2000 * i;
      }
    }
  }

  return returnVal;
}
