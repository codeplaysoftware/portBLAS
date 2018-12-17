#include <benchmark/benchmark.h>

unsigned int fibonacci(unsigned int n) {
  if (n == 0 || n == 1) return 1;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

static void BM_Fibonacci(benchmark::State& state) {
  for (auto _ : state) {
    fibonacci(state.range(0));
  };
}
// Register the function as a benchmark
BENCHMARK(BM_Fibonacci)->RangeMultiplier(2)->Range(2, 32);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state) std::string copy(x);
}
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();