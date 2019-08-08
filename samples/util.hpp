#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename index_t, typename matrix_t>
static inline void print_matrix(const matrix_t& M, index_t rows, index_t cols,
                                index_t ld) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::string print_friendly = std::to_string(M[j * ld + i]).substr(0, 6);
      std::cout << print_friendly << ((j < cols - 1) ? ' ' : '\n');
    }
  }
}

template <typename index_t, typename matrix_t>
static inline void fill_matrix(matrix_t& M, index_t rows, index_t cols,
                               index_t ld) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      M[j * ld + i] = dis(gen);
    }
  }
}

template <typename index_t, typename vector_t>
static inline void print_vector(const vector_t& V, index_t size, index_t inc) {
  for (int i = 0; i < (size - 1) * inc + 1; i += inc) {
    std::string print_friendly = std::to_string(V[i]).substr(0, 6);
    std::cout << print_friendly << '\n';
  }
}

template <typename index_t, typename vector_t>
static inline void fill_vector(vector_t& V, index_t size, index_t inc) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  for (int i = 0; i < (size - 1) * inc + 1; i += inc) {
    V[i] = dis(gen);
  }
}
