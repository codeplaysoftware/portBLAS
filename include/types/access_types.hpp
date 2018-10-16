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
 *  @filename access_patterns.hpp
 *
 **************************************************************************/

#ifndef ACCESS_TYPES_HPP
#define ACCESS_TYPES_HPP

#include "transposition_types.hpp"
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
enum class Layout { RowMajor, ColMajor };
  const Layout layout;

  static inline Layout layout_from_transposition(Transposition &t) { 
     if (t.isNormal()) {
      return Layout::RowMajor;
    } else {
      return Layout::ColMajor;
    }
  }

  static inline Layout combined_layouts(Access &a, Access &b) { 
    // In most situations, a is a device, and b is the data
    // in some cases, a is some data of layout, while b is how we want to access it. 
    // if it's not true that (device is row major  xor  data is row major), then row major
    if(a.isRowMajor() && b.isRowMajor()) { 
      // We can be row major, of course
      return Layout::RowMajor;
    }else if(a.isRowMajor() && b.isColMajor()) { 
      // We should be col major?
      return Layout::ColMajor; 
    } else if(a.isColMajor() && b.isRowMajor()) {
      // We should be col major? 
      return Layout::ColMajor; 
    } else { // a.isColMajor() && b.isColMajor()
      // We can be row major (according to Jose)
      return Layout::RowMajor; 
    }
  }

 public:
  Access(Layout l) : layout(l) {}
  Access(Transposition &t) : layout(layout_from_transposition(t)) {
   
  }

  Access(Access &a, Access &b) : layout(combined_layouts(a,b)){
    
  }

  static Access RowMajor() {
    return Access(Layout::RowMajor);
  }

  static Access ColMajor() {
    return Access(Layout::ColMajor);
  }

  const inline bool isRowMajor() const { return (layout == Layout::RowMajor); }
  const inline bool isColMajor() const { return (layout == Layout::ColMajor); }
};

#endif  // ACCESS_TYPES_HPP