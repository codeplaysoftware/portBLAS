#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  int seed = time(NULL) / 30;
  srand(seed);
  std::cout << "seed: " << seed << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
