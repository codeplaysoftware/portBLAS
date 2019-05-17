
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <thread>

#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Validate.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Tensor.h>

#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;
std::ofstream out("neon_result.txt");

void cl_arm_gem_func(int M, int K, int N, int num_iter) {
  NEGEMM arm_gemm;
  Tensor arm_a, arm_b, arm_c;
  // const TensorShape shape_a(M, K), shape_b(K, N), shape_c(M, N);
  const TensorShape shape_a(K, M), shape_b(N, K), shape_c(N, M);

  arm_a.allocator()->init(TensorInfo(shape_a, 1, DataType::F32));
  arm_b.allocator()->init(TensorInfo(shape_b, 1, DataType::F32));
  arm_c.allocator()->init(TensorInfo(shape_c, 1, DataType::F32));

  arm_gemm.configure(&arm_a, &arm_b, nullptr, &arm_c, 1.0f, 1.0f);

  arm_a.allocator()->allocate();
  arm_b.allocator()->allocate();
  arm_c.allocator()->allocate();

  fill_random_tensor(arm_a, -1.f, 1.f);
  fill_random_tensor(arm_b, -1.f, 1.f);
  fill_random_tensor(arm_c, -1.f, 1.f);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  arm_gemm.run();
  start = std::chrono::system_clock::now();
  for (int i = 0; i < num_iter; i++) {
    arm_gemm.run();
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << " GFLOP/s : "
            << (static_cast<double>((2.0 * M * N * K * num_iter)) /
                elapsed_seconds.count()) *
                   1e-9
            << "\n";

  out << M << "," << N << "," << K
      << ","
      //       << (static_cast<double>((2.0 * M * N * K * num_iter))/
      //       elapsed_seconds.count()) * 1e-9 << ", "
      << elapsed_seconds.count() / num_iter << "\n";
  //
  // arm_c.map(true);
  // Window C_window;
  // C_window.use_tensor_dimensions(arm_c.info(), Window::DimY);
  // Iterator C_it(&arm_c, C_window);
  // execute_window_loop(C_window,
  //                     [&](const Coordinates &id) {
  //                       memcpy(C + id.y() * N, C_it.ptr(), N *
  //                       sizeof(float));
  //                     },
  //                     C_it);
  // arm_c.unmap();
  arm_a.allocator()->free();
  arm_b.allocator()->free();
  arm_c.allocator()->free();
  //
  // delete[] A;
  // delete[] B;
  // delete[] C;
}

int main(int argc, char **argv) {
  for (int m = 64; m <= 1024; m *= 2) {
    for (int k = 64; k <= 1024; k *= 2) {
      for (int n = 64; n <= 1024; n *= 2) {
        cl_arm_gem_func(m, k, n, 10);
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }
  }
  return 0;
}
