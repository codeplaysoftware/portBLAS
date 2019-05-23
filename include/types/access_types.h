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
 *  @filename access_types.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_ACCESS_TYPES_H
#define SYCL_BLAS_ACCESS_TYPES_H

#include "transposition_types.h"
#include <stdexcept>

/**
 * @class Access
 * @brief A wrapper type for Layout, providing common functionality and a safer
 * interface.
 */
class Access {
 private:
  /**
   * @enum Layout
   * @brief The possible layout options for a matrix, expressed algebraically.
   */
  enum class Layout { row_major, col_major };
  const Layout layout;

  static inline Layout layout_from_transposition(Transposition &t) {
    if (t.is_normal()) {
      return Layout::row_major;
    } else {
      return Layout::col_major;
    }
  }

  static inline Layout combined_layouts(Access &a, Access &b) {
    // In most situations, a is a device, and b is the data
    // in some cases, a is some data of layout, while b is how we want to access
    // it. if it's not true that (device is row major  xor  data is row major),
    // then row major
    if (a.is_row_major() && b.is_row_major()) {
      // We can be row major, of course
      return Layout::row_major;
    } else if (a.is_row_major() && b.is_col_major()) {
      // We should be col major?
      return Layout::col_major;
    } else if (a.is_col_major() && b.is_row_major()) {
      // We should be col major?
      return Layout::col_major;
    } else {  // a.is_col_major() && b.is_col_major()
      // We can be row major (according to Jose)
      return Layout::row_major;
    }
  }

 public:
  Access(Layout l) : layout(l) {}
  Access(Transposition &t) : layout(layout_from_transposition(t)) {}

  Access(Access &a, Access &b) : layout(combined_layouts(a, b)) {}

  static Access row_major() { return Access(Layout::row_major); }

  static Access col_major() { return Access(Layout::col_major); }

  const inline bool is_row_major() const {
    return (layout == Layout::row_major);
  }
  const inline bool is_col_major() const {
    return (layout == Layout::col_major);
  }
};

#endif  // ACCESS_TYPES_H
