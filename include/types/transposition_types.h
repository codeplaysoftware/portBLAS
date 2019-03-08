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
 *  @filename transpositions_types.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_TRANSPOSITION_TYPES_H
#define SYCL_BLAS_TRANSPOSITION_TYPES_H

#include <stdexcept>

/**
 * @enum Trans
 * @brief The possible transposition options for a matrix, expressed
 * algebraically.
 */
enum class Trans : char { Normal = 'n', Transposed = 't', Conjugate = 'c' };

/**
 * @class Transposition
 * @brief A simple datatype encapsulating transposition information in a
 * semantically rich way for matrices in the blas, instead of passing
 * around/checking characters everywhere.
 */
class Transposition {
 private:
  Trans trans_;

 public:
  Transposition(char x) {
    char lx = tolower(x);
    switch (lx) {
      case 'n':
        trans_ = Trans::Normal;
        break;
      case 't':
        trans_ = Trans::Transposed;
        break;
      case 'c':
        trans_ = Trans::Conjugate;
        break;
      default:
        throw std::invalid_argument("Invalid transposition argument");
    }
  }

  Trans& get() { return trans_; }

  bool is_normal() { return (trans_ == Trans::Normal); }
  bool is_transposed() { return (trans_ == Trans::Transposed); }
  bool is_conjugate() { return (trans_ == Trans::Conjugate); }
};

#endif  // TRANSPOSITION_TYPES_H
