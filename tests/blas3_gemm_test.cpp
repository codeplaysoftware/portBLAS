#include <iostream>

#include <interface/blas3_interface_sycl_gemm.hpp>

using namespace cl::sycl;
using namespace blas;

using T = float;

int main() {
  size_t m=4096,n=4096,k=4096;
  T *ma = new T[m*n];
  T *mb = new T[n*k];
  T *mc = new T[m*k];

  std::cout << "initialized data" << std::endl;

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
    buffer<T, 1> mba(ma, range<1>{m*k});
    buffer<T, 1> mbb(mb, range<1>{k*n});
    buffer<T, 1> mbc(mc, range<1>{m*n});
    bool _tra=false,_trb=false;
    T alpha=.54,beta=.39;
    std::cout << "running kernel" << std::endl;
    _gemm(ex,_tra,_trb,m,k,n,alpha,
          mba,m,
          mbb,n,
          beta,
          mbc,k);
  }
  std::cout << "finish" << std::endl;

  delete [] ma;
  delete [] mb;
  delete [] mc;
}
