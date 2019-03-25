/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename range.hpp
 *
 **************************************************************************/

#ifndef RANGE_HPP
#define RANGE_HPP

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

/**
 * Abstract base class for a range. Define three operations that any range must
 * implement, as well as an element type:
 *  T yield()        --  Return the current value of the range, and
 *                       increment the internal value (i.e. i++)
 *  T peek()         --  Return the current value of the range, but do not
 *                       increment the internal value.
 *  bool finished()  --  Report whether the previous yield caused the range
 *                       to "finish", or rollover and reset
 *                       whatever caused it to rollover.
 */
template <typename T>
class Range {
 public:
  typedef T elem_t;
  virtual T yield() = 0;
  virtual T peek() = 0;
  virtual bool finished() = 0;
};

/**
 * Class of ranges that iterate over numeric sizes.
 */
template <typename T>
class SizeRange : public Range<T> {
  bool _finished = false;
  T v;
  T low;
  T high;
  T mult;

 public:
  typedef T elem_t;
  SizeRange(T _low, T _high, T _mult)
      : v(_low), low(_low), high(_high), mult(_mult) {}
  SizeRange(const SizeRange& sr)
      : v(sr.v), low(sr.low), high(sr.high), mult(sr.mult) {}
  T peek() { return v; }
  T yield() {
    T r = v;    // cache the current value
    v *= mult;  // increment
    return r;   // return
  }
  bool finished() {
    // Ranges are *inclusive*
    bool _finished = v > high;
    // act like a resettable latch - this allows us to iterate over a range
    // again once we've finished it - particularly useful for multidimensional
    // ranges
    if (_finished) {
      v = low;
    }
    return _finished;
  }
};

/**
 * Class of ranges that iterate over a list of concrete values.
 */
template <typename T>
class ValueRange : public Range<T> {
  std::vector<T> vals;
  typename std::vector<T>::iterator iter;

 public:
  typedef T elem_t;
  ValueRange(std::vector<T> _vals) : vals(_vals), iter(vals.begin()) {}
  ValueRange(std::initializer_list<T> l) : vals(l), iter(vals.begin()) {}
  ValueRange(const ValueRange& vr) : vals(vr.vals), iter(vals.begin()) {}

  T peek() { return *iter; }
  T yield() { return *iter++; }
  bool finished() {
    bool _finished = iter >= vals.end();
    // act like a resettable latch - this allows us to iterate over a range
    // again once we've finished it - particularly useful for multidimensional
    // ranges
    if (_finished) {
      iter = vals.begin();
    }
    return _finished;
  }
};

/**
 * Class of ranges where we concatenate two ranges.
 */
template <typename Range1, typename Range2>
class ConcatRange : public Range<typename Range1::elem_t> {
  Range1 r1;
  Range2 r2;

  // cache this - as otherwise it'll reset!
  bool r1_finished = false;

 public:
  typedef typename Range1::elem_t elem_t;
  ConcatRange(Range1 _r1, Range2 _r2) : r1(_r1), r2(_r2){};

  elem_t peek() { return r1_finished ? r2.peek() : r1.peek(); }

  // override the behaviour of r1 - we don't want it to act as a
  // resettable latch - we want it to, instead, stop when it's done
  // so that we can move on to r2.
  elem_t yield() {
    if (r1_finished) {
      return r2.yield();
    } else {
      auto v = r1.yield();
      r1_finished = r1.finished();
      return v;
    }
  }

  // Overall, though, we *do* want to look like a resettable latch,
  // so reset our r1_finished value so we can start again.
  bool finished() {
    if (r2.finished()) {
      r1_finished = false;
      return true;
    } else {
      return false;
    }
  }
};

/**
 * Cartesian products of ranges...
 */
template <typename Range1, typename Range2>
class Range2D
    : public Range<
          std::tuple<typename Range1::elem_t, typename Range2::elem_t>> {
  Range1 r1;
  Range2 r2;

 public:
  typedef std::tuple<typename Range1::elem_t, typename Range2::elem_t> elem_t;
  Range2D(Range1 _r1, Range2 _r2) : r1(_r1), r2(_r2) {}
  elem_t peek() { return std::make_tuple(r1.peek(), r2.peek()); }
  elem_t yield() {
    auto rv2 = r2.yield();
    auto rv1 = r2.finished() ? r1.yield() : r1.peek();
    return std::make_tuple(rv1, rv2);
  }
  bool finished() { return r1.finished(); }
};

template <typename Range1, typename Range2, typename Range3>
class Range3D
    : public Range<std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                              typename Range3::elem_t>> {
  Range1 r1;
  Range2 r2;
  Range3 r3;

 public:
  typedef std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                     typename Range3::elem_t>
      elem_t;
  Range3D(Range1 _r1, Range2 _r2, Range3 _r3) : r1(_r1), r2(_r2), r3(_r3) {}
  elem_t peek() { return std::make_tuple(r1.peek(), r2.peek(), r3.peek()); }
  elem_t yield() {
    auto rv3 = r3.yield();
    auto rv2 = r3.finished() ? r2.yield() : r2.peek();
    auto rv1 = r2.finished() ? r1.yield() : r1.peek();
    return std::make_tuple(rv1, rv2, rv3);
  }
  bool finished() { return r1.finished(); }
};

template <typename Range1, typename Range2, typename Range3, typename Range4>
class Range4D
    : public Range<
          std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                     typename Range3::elem_t, typename Range4::elem_t>> {
  Range1 r1;
  Range2 r2;
  Range3 r3;
  Range4 r4;

 public:
  typedef std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                     typename Range3::elem_t, typename Range4::elem_t>
      elem_t;
  Range4D(Range1 _r1, Range2 _r2, Range3 _r3, Range4 _r4)
      : r1(_r1), r2(_r2), r3(_r3), r4(_r4) {}
  elem_t peek() {
    return std::make_tuple(r1.peek(), r2.peek(), r3.peek(), r4.peek());
  }
  elem_t yield() {
    auto rv4 = r4.yield();
    auto rv3 = r4.finished() ? r3.yield() : r3.peek();
    auto rv2 = r3.finished() ? r2.yield() : r2.peek();
    auto rv1 = r2.finished() ? r1.yield() : r1.peek();
    return std::make_tuple(rv1, rv2, rv3, rv4);
  }
  bool finished() { return r1.finished(); }
};

template <typename Range1, typename Range2, typename Range3, typename Range4,
          typename Range5>
class Range5D
    : public Range<std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                              typename Range3::elem_t, typename Range4::elem_t,
                              typename Range5::elem_t>> {
  Range1 r1;
  Range2 r2;
  Range3 r3;
  Range4 r4;
  Range5 r5;

 public:
  typedef std::tuple<typename Range1::elem_t, typename Range2::elem_t,
                     typename Range3::elem_t, typename Range4::elem_t,
                     typename Range5::elem_t>
      elem_t;
  Range5D(Range1 _r1, Range2 _r2, Range3 _r3, Range4 _r4, Range5 _r5)
      : r1(_r1), r2(_r2), r3(_r3), r4(_r4), r5(_r5) {}
  elem_t peek() {
    return std::make_tuple(r1.peek(), r2.peek(), r3.peek(), r4.peek(),
                           r5.peek());
  }
  elem_t yield() {
    auto rv5 = r5.yield();
    auto rv4 = r5.finished() ? r4.yield() : r4.peek();
    auto rv3 = r4.finished() ? r3.yield() : r3.peek();
    auto rv2 = r3.finished() ? r2.yield() : r2.peek();
    auto rv1 = r2.finished() ? r1.yield() : r1.peek();
    return std::make_tuple(rv1, rv2, rv3, rv4, rv5);
  }
  bool finished() { return r1.finished(); }
};

/**
 * Utility range constructors.
 */
template <typename T>
SizeRange<T> size_range(T low, T high, T mult) {
  return SizeRange<T>(low, high, mult);
}

template <typename Vect>
ValueRange<typename Vect::value_type> value_range(Vect vals) {
  return ValueRange<typename Vect::value_type>(vals);
}

template <typename T>
ValueRange<T> value_range(std::initializer_list<T> l) {
  return ValueRange<T>(l);
}

template <typename Range1, typename Range2>
Range2D<Range1, Range2> nd_range(Range1 r1, Range2 r2) {
  return Range2D<Range1, Range2>(r1, r2);
}

template <typename Range1, typename Range2, typename Range3>
Range3D<Range1, Range2, Range3> nd_range(Range1 r1, Range2 r2, Range3 r3) {
  return Range3D<Range1, Range2, Range3>(r1, r2, r3);
}

template <typename Range1, typename Range2, typename Range3, typename Range4>
Range4D<Range1, Range2, Range3, Range4> nd_range(Range1 r1, Range2 r2,
                                                 Range3 r3, Range4 r4) {
  return Range4D<Range1, Range2, Range3, Range4>(r1, r2, r3, r4);
}

template <typename Range1, typename Range2, typename Range3, typename Range4,
          typename Range5>
Range5D<Range1, Range2, Range3, Range4, Range5> nd_range(Range1 r1, Range2 r2,
                                                         Range3 r3, Range4 r4,
                                                         Range5 r5) {
  return Range5D<Range1, Range2, Range3, Range4, Range5>(r1, r2, r3, r4, r5);
}

template <typename Range1, typename Range2>
ConcatRange<Range1, Range2> concat_ranges(Range1 r1, Range2 r2) {
  return ConcatRange<Range1, Range2>(r1, r2);
}

namespace default_ranges {

auto level_1 = size_range(1 << 1, 1 << 24, 1 << 1);

auto level_2 = nd_range(value_range({"n", "t"}), size_range(128, 4096, 2),
                        size_range(128, 4096, 2));

auto level_3 =
    concat_ranges(nd_range(value_range({"n", "t"}), value_range({"n", "t"}),
                           size_range(512, 4096, 2), size_range(512, 4096, 2),
                           size_range(512, 4096, 2)),
                  value_range({std::make_tuple("n", "n", 8192, 128, 8192),
                               std::make_tuple("t", "n", 8192, 128, 8192),
                               std::make_tuple("n", "t", 8192, 128, 8192)}));

}  // namespace default_ranges

#endif  // include guard
