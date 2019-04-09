#ifndef CLI_ARGS_HPP
#define CLI_ARGS_HPP

#include <string>

#include "argparse.hpp"

namespace blas_benchmark {

typedef struct {
  std::string program_name;
  std::string device;
  std::string csv_param;
} Args;

namespace utils {

/**
 * @fn parse_args
 * @brief Returns a structure containing the data extracted from the
 * command-line arguments.
 */
inline Args parse_args(int argc, char** argv) {
  ArgumentParser parser;
  parser.addArgument("--help", 0, true);
  parser.addArgument("--device", 1, true);
  parser.addArgument("--csv_param", 1, false);

  parser.parse(argc, const_cast<const char**>(argv));

  Args args = {
    argv[0],
    parser.retrieve<std::string>("device"),
    parser.retrieve<std::string>("csv_param")
  };

  return args;
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
