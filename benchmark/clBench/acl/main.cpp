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

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &success);

  // Run the benchmarks
  benchmark::RunSpecifiedBenchmarks();

  return !success;
}
