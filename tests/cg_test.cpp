#include <algorithm>
#include <cstdlib>
#include <interface/blas1_interface_sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cl::sycl;
using namespace blas;

#define DEF_SIZE_VECT 1200
#define MAX_ITR_LOOP 1000
#define ERROR_ALLOWED 1.0E-8

#define RANDOM_DATA 1

// This routine assures the symmetric matrix defined by td ans ts is SPD
void trdSP(std::vector<double> &td, std::vector<double> &ts) {
  int size = td.size();

  for (int i = 0; i < size; i++) {
    td[i] += (i > 0) ? ts[i - 1] : 0.0;
    td[i] += (i + 1 < size) ? ts[i] : 0.0;
  }
}

//
template <typename ExecutorType, typename T, typename ContainerT>
void prdTrdSP(Executor<ExecutorType> ex, vector_view<T, ContainerT> &td,
              vector_view<T, ContainerT> &ts, vector_view<T, ContainerT> &x,
              vector_view<T, ContainerT> &y) {
  size_t size = x.getSize();
  auto my_td0 = vector_view<T, ContainerT>(td, 0, 1, size);
  auto my_ts = vector_view<T, ContainerT>(ts, 0, 1, size - 1);
  auto my_x0 = vector_view<T, ContainerT>(x, 0, 1, size);
  auto my_x1 = vector_view<T, ContainerT>(x, 1, 1, size - 1);
  auto my_x2 = vector_view<T, ContainerT>(x, 0, 1, size - 1);
  auto my_y0 = vector_view<T, ContainerT>(y, 0, 1, size);
  auto my_y1 = vector_view<T, ContainerT>(y, 0, 1, size - 1);
  auto my_y2 = vector_view<T, ContainerT>(y, 1, 1, size - 1);

  // Computations related to the diagonal
  auto prdVctOp0 = make_op<BinaryOp, prdOp2_struct>(my_x0, my_td0);
  auto assignOp0 = make_op<Assign>(my_y0, prdVctOp0);
  ex.execute(assignOp0);

  // Computations related to the superdiagonal
  auto prdVctOp1 = make_op<BinaryOp, prdOp2_struct>(my_x1, my_ts);
  auto addVctOp1 = make_op<BinaryOp, addOp2_struct>(my_y1, prdVctOp1);
  auto assignOp1 = make_op<Assign>(my_y1, addVctOp1);
  ex.execute(assignOp1);

  // Computations related to the subdiagonal
  auto prdVctOp2 = make_op<BinaryOp, prdOp2_struct>(my_x2, my_ts);
  auto addVctOp2 = make_op<BinaryOp, addOp2_struct>(my_y2, prdVctOp2);
  auto assignOp2 = make_op<Assign>(my_y2, addVctOp2);
  ex.execute(assignOp2);
}

template <class RHS1, class RHS2>
struct TrdMatVctPrd {
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;

  value_type eval(size_t i) {
    auto dim = r2.getSize();
    auto val = r1.eval(i) * ((i > 0) ? r2.eval(i - 1) : 0.0) +
               r1.eval(i + dim) * r2.eval(i) +
               r1.eval(i + 2 * dim) * ((i < (dim - 1)) ? r2.eval(i + 1) : 0.0);
    return val;
  }

  int getSize() { return r2.getSize(); }

  TrdMatVctPrd(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};
};

template <class RHS1, class RHS2>
TrdMatVctPrd<RHS1, RHS2> make_trdMatVctPrd(RHS1 &r1, RHS2 &r2) {
  return TrdMatVctPrd<RHS1, RHS2>(r1, r2);
}

namespace blas {

template <typename RHS1, typename RHS2>
struct Evaluate<TrdMatVctPrd<RHS1, RHS2>> {
  using value_type = typename RHS2::value_type;
  using rhs1_type = typename Evaluate<RHS1>::type;
  using rhs2_type = typename Evaluate<RHS2>::type;
  using input_type = TrdMatVctPrd<RHS1, RHS2>;
  using type = TrdMatVctPrd<rhs1_type, rhs2_type>;

  static type convert_to(input_type v, cl::sycl::handler &h) {
    auto rhs1 = Evaluate<RHS1>::convert_to(v.r1, h);
    auto rhs2 = Evaluate<RHS2>::convert_to(v.r2, h);
    return type(rhs1, rhs2);
  }
};
}  // namespace blas

void CG_1(size_t dim, double thold, size_t maxItr, cl::sycl::queue q) {
  // Variables definition
  std::vector<double> vX(dim);
  std::vector<double> vZ(dim);
  std::vector<double> vR(dim);
  std::vector<double> vB(dim);
  std::vector<double> vD(dim);
  std::vector<double> vTD(dim);
  std::vector<double> vTS(dim);
  std::vector<double> vTT(dim * 3);
  std::vector<double> vBe(1);
  std::vector<double> vTo(1);
  std::vector<double> vAl(1);
  std::vector<double> vRh(1);

  // Initializing the data
  size_t vSeed, gap;
  double minV, maxV;

  minV = 1.00;
  maxV = 0.00;
  std::for_each(std::begin(vX), std::end(vX), [&](double &elem) {
    elem = minV;
    minV += maxV;
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 0.15;
  maxV = 0.15;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTD), std::end(vTD), [&](double &elem) {
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
  minV = 0.10;
  maxV = 0.10;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTS), std::end(vTS), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

  // Assuring the matrix is SPD
  trdSP(vTD, vTS);

  Executor<SYCL> ex(q);

  // Scalars required in the method
  double tol, alpha, rho;
  size_t step = 0;
  {
    // Parameters creation
    buffer<double, 1> bB(vB.data(), range<1>{vB.size()});
    buffer<double, 1> bX(vX.data(), range<1>{vX.size()});
    buffer<double, 1> bZ(vZ.data(), range<1>{vZ.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<double, 1> bD(vD.data(), range<1>{vD.size()});
    buffer<double, 1> bTD(vTD.data(), range<1>{vTD.size()});
    buffer<double, 1> bTS(vTS.data(), range<1>{vTS.size()});
    buffer<double, 1> bTT(vTT.data(), range<1>{vTT.size()});
    buffer<double, 1> bBe(vBe.data(), range<1>{vBe.size()});
    buffer<double, 1> bTo(vTo.data(), range<1>{vTo.size()});
    buffer<double, 1> bAl(vAl.data(), range<1>{vAl.size()});
    buffer<double, 1> bRh(vRh.data(), range<1>{vRh.size()});

    // Construct the view of the Buffers
    BufferVectorView<double> bvB(bB);
    BufferVectorView<double> bvX(bX);
    BufferVectorView<double> bvZ(bZ);
    BufferVectorView<double> bvR(bR);
    BufferVectorView<double> bvD(bD);
    BufferVectorView<double> bvTD(bTD);
    BufferVectorView<double> bvTS(bTS);
    BufferVectorView<double> bvTT(bTT);
    BufferVectorView<double> bvBe(bBe);
    BufferVectorView<double> bvTo(bTo);
    BufferVectorView<double> bvAl(bAl);
    BufferVectorView<double> bvRh(bRh);

    // Computation of the RHS related to x=[1*];
    prdTrdSP<SYCL>(ex, bvTD, bvTS, bvX, bvB);  // b = A * x

    // Creating the tridiagonal matrix from the vectors
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 1, 1);
    _copy<SYCL>(ex, bvTD.getSize(), bvTD, 1, bvTT + dim, 1);
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 2 * dim, 1);

    {
      auto initOp = make_op<UnaryOp, iniAddOp1_struct>(bvX);
      auto assignOp = make_op<Assign>(bvX, initOp);
      ex.execute(assignOp);  // x = 0
    }

    // Begin CG

    //    prdTrdSP<SYCL>(ex, bvTD, bvTS, bvX, bvZ);             // z = A * x
    {
      auto prdMtrVctOp = make_trdMatVctPrd(bvTT, bvX);
      auto assignOp = make_op<Assign>(bvZ, prdMtrVctOp);
      ex.execute(assignOp);
    }

    _copy<SYCL>(ex, dim, bvB + 0, 1, bvR + 0, 1);        // r = b
    _axpy<SYCL>(ex, dim, -1.0, bvZ + 0, 1, bvR + 0, 1);  // r = r - z
    _copy<SYCL>(ex, dim, bvR + 0, 1, bvD + 0, 1);        // d = r
    _dot<SYCL>(ex, dim, bvR + 0, 1, bvR + 0, 1, bvBe);   // beta = r' * r
    {
      auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(bvBe);
      auto assignOp = make_op<Assign>(bvTo, sqrtOp);  // tol = sqrt(beta)
      ex.execute(assignOp);

      auto hostAcc =
          bTo.get_access<access::mode::read, access::target::host_buffer>();
      tol = hostAcc[0];
    }
#ifdef VERBOSE
    printf("tol = %e \n", tol);
#endif  // VERBOSE
    while ((tol > thold) && (step < maxItr)) {
      {
        auto prdMtrVctOp = make_trdMatVctPrd(bvTT, bvD);
        auto assignOp = make_op<Assign>(bvZ, prdMtrVctOp);
        ex.execute(assignOp);
      }
      _dot<SYCL>(ex, dim, bvD + 0, 1, bvZ + 0, 1, bvAl);  // alpha = d' * z
      {
        auto divOp = make_op<BinaryOp, divOp2_struct>(bvBe, bvAl);
        auto assignOp = make_op<Assign>(bvRh, divOp);  // rho = beta / alpha
        ex.execute(assignOp);

        auto hostAcc =
            bRh.get_access<access::mode::read, access::target::host_buffer>();
        rho = hostAcc[0];
      }
      _axpy<SYCL>(ex, dim, rho, bvD + 0, 1, bvX + 0, 1);   // x = x + rho * d
      _axpy<SYCL>(ex, dim, -rho, bvZ + 0, 1, bvR + 0, 1);  // r = r - rho * z
      {
        auto assignOp = make_op<Assign>(bvAl, bvBe);  // alpha = beta
        ex.execute(assignOp);
      }
      _dot<SYCL>(ex, dim, bvR + 0, 1, bvR + 0, 1, bvBe);  // beta = r' * r
      {
        auto divOp = make_op<BinaryOp, divOp2_struct>(bvBe, bvAl);
        auto assignOp = make_op<Assign>(bvAl, divOp);  // alpha = beta / alpha
        ex.execute(assignOp);

        auto hostAcc =
            bAl.get_access<access::mode::read, access::target::host_buffer>();
        alpha = hostAcc[0];
      }
      _scal<SYCL>(ex, dim, alpha, bvD + 0, 1);            // d = alpha * d
      _axpy<SYCL>(ex, dim, 1.0, bvR + 0, 1, bvD + 0, 1);  // d = d + r
      {
        auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(bvBe);
        auto assignOp = make_op<Assign>(bvTo, sqrtOp);  // tol = sqrt(beta)
        ex.execute(assignOp);

        auto hostAcc =
            bTo.get_access<access::mode::read, access::target::host_buffer>();
        tol = hostAcc[0];
      }
      step++;
#ifdef VERBOSE
      printf("tol = %e \n", tol);
#endif  // VERBOSE
    }

    // End CG
    printf("tolF = %e , steps = %ld\n", tol, step);
  }
}

template <typename ExecutorType, typename T, typename ContainerT>
void _xpay(Executor<ExecutorType> ex, int _N, double _alpha,
           vector_view<T, ContainerT> _vx, int _incx,
           vector_view<T, ContainerT> _vy, int _incy) {
  // Calculating: _vy = alpha * _vx + _vy
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, my_vy);
  auto addBinaryOp = make_op<BinaryOp, addOp2_struct>(my_vx, scalOp);
  auto assignOp = make_op<Assign>(my_vy, addBinaryOp);
  ex.execute(assignOp);
}

template <typename ExecutorType, typename T, typename ContainerT>
void _two_axpy_dotSng_Scal(Executor<ExecutorType> ex, int _N, double _alpha1,
                           vector_view<T, ContainerT> _vx1, int _incx1,
                           vector_view<T, ContainerT> _vy1, int _incy1,
                           double _alpha2, vector_view<T, ContainerT> _vx2,
                           int _incx2, vector_view<T, ContainerT> _vy2,
                           int _incy2, vector_view<T, ContainerT> _be,
                           vector_view<T, ContainerT> _al,
                           vector_view<T, ContainerT> _to, int blqS, int nBlq) {
  //  Calculating: _vy1 = _alpha1 * _vx1 + _vy1
  auto my_vx1 = vector_view<T, ContainerT>(_vx1, _vx1.getDisp(), _incx1, _N);
  auto my_vy1 = vector_view<T, ContainerT>(_vy1, _vy1.getDisp(), _incy1, _N);
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_alpha1, my_vx1);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(my_vy1, scalOp1);

  //  Calculating: _vy2 = _alpha2 * _vx2 + _vy2
  auto my_vx2 = vector_view<T, ContainerT>(_vx2, _vx2.getDisp(), _incx2, _N);
  auto my_vy2 = vector_view<T, ContainerT>(_vy2, _vy2.getDisp(), _incy2, _N);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha2, my_vx2);
  auto addBinaryOp2 = make_op<BinaryOp, addOp2_struct>(my_vy2, scalOp2);

  // join both operations
  auto assignOp1 = make_op<Assign>(my_vy1, addBinaryOp1);
  auto assignOp2 = make_op<Assign>(my_vy2, addBinaryOp2);
  auto dobleAssignOp = make_op<Join>(assignOp1, assignOp2);

  //  Calculating: _vy2 .* _vy2
  auto prodOp = make_op<UnaryOp, prdOp1_struct>(dobleAssignOp);
  auto my_be = vector_view<T, ContainerT>(_be, _be.getDisp(), 1, 1);
  auto my_al = vector_view<T, ContainerT>(_al, _al.getDisp(), 1, 1);
  auto my_to = vector_view<T, ContainerT>(_to, _to.getDisp(), 1, 1);
  auto assignOp3 = make_addAssignReduction(my_to, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp3);

  // Calculating: _al = _to / _be; _be = _to; _to = sqrt(_to);
  auto divOp = make_op<BinaryOp, divOp2_struct>(my_to, my_be);
  auto assignOp41 = make_op<Assign>(my_al, divOp);
  auto assignOp42 = make_op<Assign>(my_be, my_to);
  auto joinOp = make_op<Join>(assignOp41, assignOp42);
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(joinOp);
  auto assignOp43 = make_op<Assign>(my_to, sqrtOp);
  ex.execute(assignOp43);
}

template <typename ExecutorType, typename T, typename ContainerT>
void prdTrdSP2_dot_Scal(Executor<ExecutorType> ex, int _N,
                        vector_view<T, ContainerT> _TT, int _incTT,
                        vector_view<T, ContainerT> _vx, int _incx,
                        vector_view<T, ContainerT> _vy, int _incy,
                        vector_view<T, ContainerT> _be,
                        vector_view<T, ContainerT> _rh, int blqS, int nBlq) {
  auto my_TT = vector_view<T, ContainerT>(_TT, _TT.getDisp(), _incTT, _N);
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto my_be = vector_view<T, ContainerT>(_be, _be.getDisp(), 1, 1);
  auto my_rh = vector_view<T, ContainerT>(_rh, _rh.getDisp(), 1, 1);

  //  Calculating: _vy = _TT * _vx
  auto prdMtrVctOp = make_trdMatVctPrd(my_TT, my_vx);
  auto assignOp = make_op<Assign>(my_vy, prdMtrVctOp);

  //  Calculating: _vx .* _vy
  auto prodOp = make_op<BinaryOp, prdOp2_struct>(my_vx, assignOp);
  auto assignOp1 = make_addAssignReduction(my_rh, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp1);

  //  Calculating: _rh = _be / _rh;
  auto divOp = make_op<BinaryOp, divOp2_struct>(my_be, my_rh);
  auto assignOp2 = make_op<Assign>(my_rh, divOp);
#ifdef BLAS_EXPERIMENTAL
  ex.execute(assignOp2, 1);
#endif  // BLAS_EXPERIMENTAL
  ex.execute(assignOp2);
}

template <typename ExecutorType, typename T, typename ContainerT>
void prdTrdSP2_init_vectors_dotSng_Scal(
    Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _TT,
    int _incTT, vector_view<T, ContainerT> _vx, int _incx, double _alpha,
    vector_view<T, ContainerT> _vb, int _incb, vector_view<T, ContainerT> _vr,
    int _incr, vector_view<T, ContainerT> _vz, int _incz,
    vector_view<T, ContainerT> _vd, int _incd, vector_view<T, ContainerT> _be,
    vector_view<T, ContainerT> _to, int blqS, int nBlq) {
  auto my_TT = vector_view<T, ContainerT>(_TT, _TT.getDisp(), _incTT, _N);
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vb = vector_view<T, ContainerT>(_vb, _vb.getDisp(), _incb, _N);
  auto my_vr = vector_view<T, ContainerT>(_vr, _vr.getDisp(), _incr, _N);
  auto my_vz = vector_view<T, ContainerT>(_vz, _vz.getDisp(), _incz, _N);
  auto my_vd = vector_view<T, ContainerT>(_vd, _vd.getDisp(), _incd, _N);
  auto my_be = vector_view<T, ContainerT>(_be, _be.getDisp(), 1, 1);
  auto my_to = vector_view<T, ContainerT>(_to, _to.getDisp(), 1, 1);

  // Calculating: _vz = TT * _vx
  auto prdMtrVctOp = make_trdMatVctPrd(my_TT, my_vx);
  auto assignOp = make_op<Assign>(my_vz, prdMtrVctOp);
  // Calculating: _vd = _vr = _vb + alpha * _vz
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_alpha, assignOp);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(my_vb, scalOp1);
  auto assignOp11 = make_op<Assign>(my_vr, addBinaryOp1);
  auto assignOp12 = make_op<Assign>(my_vd, assignOp11);

  // Calculating: _vd .* _vd
  auto prodOp = make_op<UnaryOp, prdOp1_struct>(assignOp12);
  auto assignOp1 = make_addAssignReduction(my_be, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp1);

  // Calculating: _to = sqrt(be)
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(my_be);
  auto assignOp2 = make_op<Assign>(my_to, sqrtOp);
  ex.execute(assignOp2);
}

void CG_4(size_t dim, double thold, size_t maxItr, cl::sycl::queue q) {
  // Variables definition
  std::vector<double> vX(dim);
  std::vector<double> vZ(dim);
  std::vector<double> vR(dim);
  std::vector<double> vB(dim);
  std::vector<double> vD(dim);
  std::vector<double> vTD(dim);
  std::vector<double> vTS(dim);
  std::vector<double> vTT(dim * 3);
  std::vector<double> vBe(1);
  std::vector<double> vTo(1);
  std::vector<double> vAl(1);
  std::vector<double> vRh(1);

  // Initializing the data
  size_t vSeed, gap;
  double minV, maxV;

  minV = 1.00;
  maxV = 0.00;
  std::for_each(std::begin(vX), std::end(vX), [&](double &elem) {
    elem = minV;
    minV += maxV;
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 0.15;
  maxV = 0.15;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTD), std::end(vTD), [&](double &elem) {
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
  minV = 0.10;
  maxV = 0.10;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTS), std::end(vTS), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

  // Assuring the matrix is SPD
  trdSP(vTD, vTS);

  Executor<SYCL> ex(q);

  // Scalars required in the method
  double tol, alpha, rho;
  size_t step = 0;
  {
    // Parameters creation
    buffer<double, 1> bB(vB.data(), range<1>{vB.size()});
    buffer<double, 1> bX(vX.data(), range<1>{vX.size()});
    buffer<double, 1> bZ(vZ.data(), range<1>{vZ.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<double, 1> bD(vD.data(), range<1>{vD.size()});
    buffer<double, 1> bTD(vTD.data(), range<1>{vTD.size()});
    buffer<double, 1> bTS(vTS.data(), range<1>{vTS.size()});
    buffer<double, 1> bTT(vTT.data(), range<1>{vTT.size()});
    buffer<double, 1> bBe(vBe.data(), range<1>{vBe.size()});
    buffer<double, 1> bTo(vTo.data(), range<1>{vTo.size()});
    buffer<double, 1> bAl(vAl.data(), range<1>{vAl.size()});
    buffer<double, 1> bRh(vRh.data(), range<1>{vRh.size()});

    // Construct the view of the Buffers
    BufferVectorView<double> bvB(bB);
    BufferVectorView<double> bvX(bX);
    BufferVectorView<double> bvZ(bZ);
    BufferVectorView<double> bvR(bR);
    BufferVectorView<double> bvD(bD);
    BufferVectorView<double> bvTD(bTD);
    BufferVectorView<double> bvTS(bTS);
    BufferVectorView<double> bvTT(bTT);
    BufferVectorView<double> bvBe(bBe);
    BufferVectorView<double> bvTo(bTo);
    BufferVectorView<double> bvAl(bAl);
    BufferVectorView<double> bvRh(bRh);

    // Computation of the RHS related to x=[1*];
    prdTrdSP<SYCL>(ex, bvTD, bvTS, bvX, bvB);  // b = A * x

    // Creating the tridiagonal matrix from the vectors
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 1, 1);
    _copy<SYCL>(ex, bvTD.getSize(), bvTD, 1, bvTT + dim, 1);
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 2 * dim, 1);

    auto blqS = 256;  // size of each workgroups
    auto nBlq = 512;  // number of workgorups

    {
      auto initOp = make_op<UnaryOp, iniAddOp1_struct>(bvX);
      auto assignOp = make_op<Assign>(bvX, initOp);
      ex.execute(assignOp);  // x = 0
    }

    // Begin CG

    prdTrdSP2_init_vectors_dotSng_Scal<SYCL>(ex, dim,  // z = A * x
                                             bvTT + 0, 1, bvX + 0, 1,
                                             -1.0,  // r = b - z: d = r
                                             bvB, 1, bvR, 1,  // beta = r' * r
                                             bvZ, 1, bvD,
                                             1,  // tol = sqrt(beta)
                                             bvBe, bvTo, blqS, nBlq);
    {
      auto hostAccT =
          bTo.get_access<access::mode::read, access::target::host_buffer>();
      tol = hostAccT[0];
    }
#ifdef VERBOSE
    printf("tol = %e \n", tol);
#endif
    while ((tol > thold) && (step < maxItr)) {
      prdTrdSP2_dot_Scal<SYCL>(ex, dim, bvTT + 0, 1,    // z = A * d
                               bvD + 0, 1, bvZ + 0, 1,  // alpha = d' * z
                               bvBe, bvRh,              // rho = beta / alpha
                               blqS, nBlq);
      {
        auto hostAccR =
            bRh.get_access<access::mode::read, access::target::host_buffer>();
        rho = hostAccR[0];
      }
      _two_axpy_dotSng_Scal<SYCL>(
          ex, dim, rho,                  // x = x + rho * d
          bvD + 0, 1, bvX + 0, 1, -rho,  // r = r - rho * z
          bvZ + 0, 1, bvR + 0, 1,        // alpha = beta
          bvBe, bvAl, bvTo,              // beta = r' * r; alpha = beta / alpha
          blqS, nBlq);                   // tol = sqrt (tol)
      {
        auto hostAccA =
            bAl.get_access<access::mode::read, access::target::host_buffer>();
        alpha = hostAccA[0];
        auto hostAccT =
            bTo.get_access<access::mode::read, access::target::host_buffer>();
        tol = hostAccT[0];
      }
      _xpay<SYCL>(ex, dim, alpha, bvR + 0, 1, bvD + 0, 1);  // d = alpha * d + r
      step++;
#ifdef VERBOSE
      printf("tol = %e \n", tol);
#endif  // VERBOSE
    }
    // End CG
    printf("tolF = %e , steps = %ld\n", tol, step);
  }
}

template <typename ExecutorType, typename T, typename ContainerT>
void _xpayF(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vAl,
            vector_view<T, ContainerT> _vx, vector_view<T, ContainerT> _vy) {
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_vAl, _vy);
  auto addBinaryOp = make_op<BinaryOp, addOp2_struct>(_vx, scalOp);
  auto assignOp = make_op<Assign>(_vy, addBinaryOp);
  ex.execute(assignOp);
}

template <typename ExecutorType, typename T, typename ContainerT>
void _two_axpy_dotSng_ScalF(
    Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vAl1,
    vector_view<T, ContainerT> _vx1, vector_view<T, ContainerT> _vy1,
    vector_view<T, ContainerT> _vAl2, vector_view<T, ContainerT> _vx2,
    vector_view<T, ContainerT> _vy2, vector_view<T, ContainerT> _be,
    vector_view<T, ContainerT> _al, vector_view<T, ContainerT> _to,
    bool compSqrt, int blqS, int nBlq) {
  // _vy1 = _vAl * _vx1 + _vy1
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_vAl1, _vx1);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(_vy1, scalOp1);

  // _vy2 = -_vAl * _vx2 + _vy2
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_vAl2, _vx2);
  auto addBinaryOp2 = make_op<BinaryOp, addOp2_struct>(_vy2, scalOp2);

  // join both operations
  auto assignOp1 = make_op<Assign>(_vy1, addBinaryOp1);
  auto assignOp2 = make_op<Assign>(_vy2, addBinaryOp2);
  auto dobleAssignOp = make_op<Join>(assignOp1, assignOp2);

  // _vy2 .* _vy2
  auto prodOp = make_op<UnaryOp, prdOp1_struct>(dobleAssignOp);

  // _to = reduction(_vy2 .* _vy2)
  auto assignOp3 = make_addAssignReduction(_to, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp3);

  //	_al = _to / _be; _be = _to; _to = sqrt(_to);
  auto divOp = make_op<BinaryOp, divOp2_struct>(_to, _be);
  auto assignOp41 = make_op<Assign>(_al, divOp);
  auto assignOp42 = make_op<Assign>(_be, _to);
  auto joinOp = make_op<Join>(assignOp41, assignOp42);
  if (compSqrt) {
    auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(joinOp);
    auto assignOp43 = make_op<Assign>(_to, sqrtOp);
    ex.execute(assignOp43);
  } else {
    ex.execute(joinOp);
  }
}

template <typename ExecutorType, typename T, typename ContainerT>
void prdTrdSP2_dot_ScalF(Executor<ExecutorType> ex, int _N,
                         vector_view<T, ContainerT> _TT,
                         vector_view<T, ContainerT> _vx,
                         vector_view<T, ContainerT> _vy,
                         vector_view<T, ContainerT> _be,
                         vector_view<T, ContainerT> _rh,
                         vector_view<T, ContainerT> _al, int blqS, int nBlq) {
  // _vy = _TT * _vx
  auto prdMtrVctOp = make_trdMatVctPrd(_TT, _vx);
  auto assignOp = make_op<Assign>(_vy, prdMtrVctOp);

  // _vx .* _vy
  auto prodOp = make_op<BinaryOp, prdOp2_struct>(_vx, assignOp);

  // _rh = reduction( _vx .* _vy)
  auto assignOp1 = make_addAssignReduction(_rh, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp1);

  //	_rh = _be / _rh; _al = -rh
  auto divOp = make_op<BinaryOp, divOp2_struct>(_be, _rh);
  auto assignOp2 = make_op<Assign>(_rh, divOp);
  auto negOp3 = make_op<UnaryOp, negOp1_struct>(assignOp2);
  auto assignOp3 = make_op<Assign>(_al, negOp3);
  ex.execute(assignOp3);
}

template <typename ExecutorType, typename T, typename ContainerT>
void prdTrdSP2_init_vectors_dotSng_ScalF(
    Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _TT,
    vector_view<T, ContainerT> _vx, double _alpha,
    vector_view<T, ContainerT> _vb, vector_view<T, ContainerT> _vr,
    vector_view<T, ContainerT> _vz, vector_view<T, ContainerT> _vd,
    vector_view<T, ContainerT> _be, vector_view<T, ContainerT> _to, int blqS,
    int nBlq) {
  // _vz = TT * _vx
  auto prdMtrVctOp = make_trdMatVctPrd(_TT, _vx);
  auto assignOp = make_op<Assign>(_vz, prdMtrVctOp);
  // _vd = _vr = _vb + alpha * _vz
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_alpha, assignOp);
  auto addBinaryOp1 = make_op<BinaryOp, addOp2_struct>(_vb, scalOp1);
  auto assignOp11 = make_op<Assign>(_vr, addBinaryOp1);
  auto assignOp12 = make_op<Assign>(_vd, assignOp11);

  // _vd .* _vd
  auto prodOp = make_op<UnaryOp, prdOp1_struct>(assignOp12);
  // _be = reduction( _vd .* _vd)
  auto assignOp1 = make_addAssignReduction(_be, prodOp, blqS, blqS * nBlq);
  ex.reduce(assignOp1);

  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(_be);
  auto assignOp2 = make_op<Assign>(_to, sqrtOp);
  ex.execute(assignOp2);
}

void CG_5(size_t dim, double thold, size_t maxItr, cl::sycl::queue q) {
  // Variables definition
  std::vector<double> vX(dim);
  std::vector<double> vZ(dim);
  std::vector<double> vR(dim);
  std::vector<double> vB(dim);
  std::vector<double> vD(dim);
  std::vector<double> vTD(dim);
  std::vector<double> vTS(dim);
  std::vector<double> vTT(dim * 3);
  std::vector<double> vBe(1);
  std::vector<double> vTo(1);
  std::vector<double> vAl(1);
  std::vector<double> vRh(1);

  // Initializing the data
  size_t vSeed, gap;
  double minV, maxV;

  minV = 1.00;
  maxV = 0.00;
  std::for_each(std::begin(vX), std::end(vX), [&](double &elem) {
    elem = minV;
    minV += maxV;
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   //  RANDOM_DATA
  minV = 0.15;
  maxV = 0.15;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTD), std::end(vTD), [&](double &elem) {
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
  minV = 0.10;
  maxV = 0.10;
#endif  //  RANDOM_DATA
  std::for_each(std::begin(vTS), std::end(vTS), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   //  RANDOM_DATA
    elem = minV; minV += maxV;
#endif  //  RANDOM_DATA
  });

  // Assuring the matrix is SPD
  trdSP(vTD, vTS);

  Executor<SYCL> ex(q);

  // Scalars required in the method
  double tol;
  size_t step = 0;
  {
    // Parameters creation
    buffer<double, 1> bB(vB.data(), range<1>{vB.size()});
    buffer<double, 1> bX(vX.data(), range<1>{vX.size()});
    buffer<double, 1> bZ(vZ.data(), range<1>{vZ.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<double, 1> bD(vD.data(), range<1>{vD.size()});
    buffer<double, 1> bTD(vTD.data(), range<1>{vTD.size()});
    buffer<double, 1> bTS(vTS.data(), range<1>{vTS.size()});
    buffer<double, 1> bTT(vTT.data(), range<1>{vTT.size()});
    buffer<double, 1> bBe(vBe.data(), range<1>{vBe.size()});
    buffer<double, 1> bTo(vTo.data(), range<1>{vTo.size()});
    buffer<double, 1> bAl(vAl.data(), range<1>{vAl.size()});
    buffer<double, 1> bRh(vRh.data(), range<1>{vRh.size()});

    // Construct the view of the Buffers
    BufferVectorView<double> bvB(bB);
    BufferVectorView<double> bvX(bX);
    BufferVectorView<double> bvZ(bZ);
    BufferVectorView<double> bvR(bR);
    BufferVectorView<double> bvD(bD);
    BufferVectorView<double> bvTD(bTD);
    BufferVectorView<double> bvTS(bTS);
    BufferVectorView<double> bvTT(bTT);
    BufferVectorView<double> bvBe(bBe);
    BufferVectorView<double> bvTo(bTo);
    BufferVectorView<double> bvAl(bAl);
    BufferVectorView<double> bvRh(bRh);

    //    bvTD.printH("TD"); bvTS.printH("TS");
    // Computation of the RHS related to x=[1*];
    prdTrdSP<SYCL>(ex, bvTD, bvTS, bvX, bvB);  // b = A * x

    // Creating the tridiagonal matrix from the vectors
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 1, 1);
    _copy<SYCL>(ex, bvTD.getSize(), bvTD, 1, bvTT + dim, 1);
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 2 * dim, 1);

    auto blqS = 256;  // size of each workgroups
    auto nBlq = 512;  // number of workgorups

    {
      auto initOp = make_op<UnaryOp, iniAddOp1_struct>(bvX);
      auto assignOp = make_op<Assign>(bvX, initOp);
#ifdef BLAS_EXPERIMENTAL
      ex.execute(assignOp, bvX.getSize());  // x = 0
#endif                                      // BLAS_EXPERIMENTAL
      ex.execute(assignOp);                 // x = 0
    }

    // Begin CG

    prdTrdSP2_init_vectors_dotSng_ScalF<SYCL>(ex, dim,  // z = A * x
                                              bvTT, bvX,
                                              -1.0,      // r = b - z: d = r
                                              bvB, bvR,  // beta = r' * r
                                              bvZ, bvD,  // tol = sqrt(beta)
                                              bvBe, bvTo, blqS, nBlq);
    {
      auto hostAcc =
          bTo.get_access<access::mode::read, access::target::host_buffer>();
      tol = hostAcc[0];
    }
#ifdef VERBOSE
    printf("tol = %e \n", tol);
#endif  // VERBOSE
    while ((tol > thold) && (step < maxItr)) {
      prdTrdSP2_dot_ScalF<SYCL>(ex, dim, bvTT,   // z = A * d
                                bvD, bvZ, bvBe,  // alpha = d' * z
                                bvRh, bvAl,      // rho = beta / alpha
                                blqS, nBlq);     // alpha = -rho
      _two_axpy_dotSng_ScalF<SYCL>(              // x = x + rho * d
          ex, dim, bvRh,                         // r = r - rho * z
          bvD, bvX, bvAl,                        // alpha = beta
          bvZ, bvR,                              // beta = r' * r
          bvBe, bvAl, bvTo, true,                // alpha = beta / alpha
          blqS, nBlq);                           // tol = sqrt (tol)
      _xpayF<SYCL>(ex, dim, bvAl, bvR, bvD);     // d = alpha * d + r
      step++;
      {
        auto hostAccT =
            bTo.get_access<access::mode::read, access::target::host_buffer>();
        tol = hostAccT[0];
      }
#ifdef VERBOSE
      printf("tol = %e \n", tol);
#endif  // VERBOSE
    }
    // End CG
    printf("tolF = %e , steps = %ld\n", tol, step);
  }
}

void CG_6(int dim, double thold, size_t maxItr, size_t itrLoop,
          cl::sycl::queue q) {
  // Variables definition
  std::vector<double> vX(dim);
  std::vector<double> vZ(dim);
  std::vector<double> vR(dim);
  std::vector<double> vB(dim);
  std::vector<double> vD(dim);
  std::vector<double> vTD(dim);
  std::vector<double> vTS(dim);
  std::vector<double> vTT(dim * 3);
  std::vector<double> vBe(1);
  std::vector<double> vTo(1);
  std::vector<double> vAl(1);
  std::vector<double> vRh(1);

  // Initializing the data
  size_t vSeed, gap;
  double minV, maxV;

  minV = 1.00;
  maxV = 0.00;
  std::for_each(std::begin(vX), std::end(vX), [&](double &elem) {
    elem = minV;
    minV += maxV;
  });

#ifdef RANDOM_DATA
  vSeed = 1;
  minV = -10.0;
  maxV = 10.0;
  gap = (size_t)(maxV - minV + 1);
  srand(vSeed);
#else   // RANDOM_DATA
  minV = 0.15;
  maxV = 0.15;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vTD), std::end(vTD), [&](double &elem) {
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
  minV = 0.10;
  maxV = 0.10;
#endif  // RANDOM_DATA
  std::for_each(std::begin(vTS), std::end(vTS), [&](double &elem) {
#ifdef RANDOM_DATA
    elem = minV + (double)(rand() % gap);
#else   // RANDOM_DATA
    elem = minV; minV += maxV;
#endif  // RANDOM_DATA
  });

  // Assuring the matrix is SPD
  trdSP(vTD, vTS);

  Executor<SYCL> ex(q);

  // Scalars required in the method
  double tol;
  size_t step = 0;
  {
    // Parameters creation
    buffer<double, 1> bB(vB.data(), range<1>{vB.size()});
    buffer<double, 1> bX(vX.data(), range<1>{vX.size()});
    buffer<double, 1> bZ(vZ.data(), range<1>{vZ.size()});
    buffer<double, 1> bR(vR.data(), range<1>{vR.size()});
    buffer<double, 1> bD(vD.data(), range<1>{vD.size()});
    buffer<double, 1> bTD(vTD.data(), range<1>{vTD.size()});
    buffer<double, 1> bTS(vTS.data(), range<1>{vTS.size()});
    buffer<double, 1> bTT(vTT.data(), range<1>{vTT.size()});
    buffer<double, 1> bBe(vBe.data(), range<1>{vBe.size()});
    buffer<double, 1> bTo(vTo.data(), range<1>{vTo.size()});
    buffer<double, 1> bAl(vAl.data(), range<1>{vAl.size()});
    buffer<double, 1> bRh(vRh.data(), range<1>{vRh.size()});

    // Construct the view of the Buffers
    BufferVectorView<double> bvB(bB);
    BufferVectorView<double> bvX(bX);
    BufferVectorView<double> bvZ(bZ);
    BufferVectorView<double> bvR(bR);
    BufferVectorView<double> bvD(bD);
    BufferVectorView<double> bvTD(bTD);
    BufferVectorView<double> bvTS(bTS);
    BufferVectorView<double> bvTT(bTT);
    BufferVectorView<double> bvBe(bBe);
    BufferVectorView<double> bvTo(bTo);
    BufferVectorView<double> bvAl(bAl);
    BufferVectorView<double> bvRh(bRh);

    prdTrdSP<SYCL>(ex, bvTD, bvTS, bvX, bvB);  // b = A * x

    // Creating the tridiagonal matrix from the vectors
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 1, 1);
    _copy<SYCL>(ex, bvTD.getSize(), bvTD, 1, bvTT + dim, 1);
    _copy<SYCL>(ex, bvTD.getSize() - 1, bvTS, 1, bvTT + 2 * dim, 1);

    auto blqS = 256;  // size of each workgroups
    auto nBlq = 512;  // number of workgorups

    {
      auto initOp = make_op<UnaryOp, iniAddOp1_struct>(bvX);
      auto assignOp = make_op<Assign>(bvX, initOp);
      ex.execute(assignOp);  // x = 0
    }

    // Begin CG

    prdTrdSP2_init_vectors_dotSng_ScalF<SYCL>(ex, dim,  // z = A * x
                                              bvTT, bvX,
                                              -1.0,      // r = b - z: d = r
                                              bvB, bvR,  // beta = r' * r
                                              bvZ, bvD,  // tol = sqrt(beta)
                                              bvBe, bvTo, blqS, nBlq);
    {
      auto hostAcc =
          bTo.get_access<access::mode::read, access::target::host_buffer>();
      tol = hostAcc[0];
    }
#ifdef VERBOSE
    printf("tol = %e \n", tol);
#endif  // RANDOM_DATA
    while ((tol > thold) && (step < maxItr)) {
      for (size_t i = itrLoop; i > 0; i--) {
        prdTrdSP2_dot_ScalF<SYCL>(ex, dim, bvTT,   // z = A * d
                                  bvD, bvZ, bvBe,  // alpha = d' * z
                                  bvRh, bvAl,      // rho = beta / alpha
                                  blqS, nBlq);     // alpha = -rho
        _two_axpy_dotSng_ScalF<SYCL>(              // x = x + rho * d
            ex, dim, bvRh,                         // r = r - rho * z
            bvD, bvX, bvAl,                        // alpha = beta
            bvZ, bvR,                              // beta = r' * r
            bvBe, bvAl, bvTo, (i == 1),            // alpha = beta / alpha
            blqS, nBlq);                           // tol = sqrt (tol)
        _xpayF<SYCL>(ex, dim, bvAl, bvR, bvD);     // d = alpha * d + r
        step++;
      }
      {
        auto hostAccT =
            bTo.get_access<access::mode::read, access::target::host_buffer>();
        tol = hostAccT[0];
      }
#ifdef VERBOSE
      printf("tol = %e \n", tol);
#endif
    }
    // End CG
    printf("tolF = %e , steps = %ld\n", tol, step);
  }
}

int main(int argc, char *argv[]) {
  size_t sizeV;

  if (argc == 1) {
    sizeV = DEF_SIZE_VECT;
  } else if (argc == 2) {
    sizeV = atoi(argv[1]);
  } else {
    std::cout << "ERROR!! --> Incorrect number of input parameters"
              << std::endl;
    return 1;
  }

  // Definition of the queue and the executor
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

  CG_1(sizeV, ERROR_ALLOWED, MAX_ITR_LOOP, q);  // Standard implementattion

  q.wait_and_throw();

  CG_4(sizeV, ERROR_ALLOWED, MAX_ITR_LOOP, q);  // Fused version, scalars on CPU

  q.wait_and_throw();

  CG_5(sizeV, ERROR_ALLOWED, MAX_ITR_LOOP, q);  // Fused version, scalars on GPU

  q.wait_and_throw();

  CG_6(sizeV, ERROR_ALLOWED, MAX_ITR_LOOP, 5, q);  // Fused version and loop(5)

  q.wait_and_throw();

  CG_6(sizeV, ERROR_ALLOWED, MAX_ITR_LOOP, 10,
       q);  // Fused version and loop(10)

  q.wait_and_throw();

  return 0;
}
