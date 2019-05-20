#include "utils.hpp"

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  // Create the benchmarks
  blas_benchmark::create_benchmark(args);

  benchmark::RunSpecifiedBenchmarks();
}
