#include "utils.hpp"

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

#ifdef ACL_BACKEND_NEON
  std::cerr << "ACL backend: NEON" << std::endl;
#else
  std::cerr << "ACL backend: OpenCL" << std::endl;
#endif

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  // Create the benchmarks
  blas_benchmark::create_benchmark(args);

  benchmark::RunSpecifiedBenchmarks();
}
