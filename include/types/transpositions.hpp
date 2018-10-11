/***************************************************************************
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename transpositions.hpp
 *
 **************************************************************************/

#include <stdexcept>

/**
 * @enum NTC
 * @brief The possible transposition options for a matrix, expressed
 * algebraically.
 */
enum class NTC : char { Normal = 'n', Transposed = 't', Conjugate = 'c' };

/**
 * @class Transposition
 * @brief A simple datatype encapsulating transposition information in a
 * semantically rich way for matrices in the blas, instead of passing
 * around/checking characters everywhere.
 */
class Transposition {
 private:
  NTC trans;

 public:
  Transposition(char x) {
    char lx = tolower(x);
    switch (lx) {
      case 'n':
        trans = NTC::Normal;
        break;
      case 't':
        trans = NTC::Transposed;
        break;
      case 'c':
        trans = NTC::Conjugate;
        break;
      default:
        throw std::invalid_argument("Invalid transposition argument");
    }
  }

  NTC& get() { return trans; }

  bool isNormal() { return (trans == NTC::Normal); }
  bool isTransposed() { return (trans == NTC::Transposed); }
  bool isConjugate() { return (trans == NTC::Conjugate); }
};