#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int seed = 12345678;
  srand(seed);
  /* srand(time(NULL)); */
  /* std::cout << "seed: " << seed << std::endl; */
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
