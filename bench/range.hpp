#ifndef RANGE_HPP
#define RANGE_HPP

#include <algorithm>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>
#include <initializer_list>

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
template <typename T> class Range {
public:
  typedef T elem_t;
  virtual T yield() = 0;
  virtual T peek() = 0;
  virtual bool finished() = 0;
};

/**
 * Class of ranges that iterate over numeric sizes.
 */
template <typename T> class SizeRange : public Range<T> {
  bool _finished = false;
  T v;
  T low;
  T high;
  T mult;

public:
  typedef T elem_t;
  SizeRange(T _low, T _high, T _mult)
      : v(_low), low(_low), high(_high), mult(_mult) {}
  T peek() { return v; }
  T yield() {
    T r = v;   // cache the current value
    v *= mult; // increment
    return r;  // return
  }
  bool finished() {
    bool _finished = v > high;
    if (_finished) {
      v = low;
    }
    return _finished;
  }
};

/**
 * Class of ranges that iterate over a list of concrete values.
 */
template <typename T> class ValueRange : public Range<T> {
  std::vector<T> vals;
  typename std::vector<T>::iterator iter;

public:
  typedef T elem_t;
  ValueRange(std::vector<T> _vals) : vals(_vals), iter(vals.begin()) {}
  ValueRange(std::initializer_list<T> l) : vals(l), iter(vals.begin()) {}
  T peek() { return *iter; }
  T yield() { return *iter++; }
  bool finished() {
    bool _finished = iter >= vals.end();
    if (_finished) {
      iter = vals.begin();
    }
    return _finished;
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

/**
 * Utility range constructors.
 */

template <typename T> SizeRange<T> range(T low, T high, T mult) {
  return SizeRange<T>(low, high, mult);
}

template <typename Vect>
ValueRange<typename Vect::value_type> range(Vect vals) {
  return ValueRange<typename Vect::value_type>(vals);
}

template <typename T> 
ValueRange<T> range(std::initializer_list<T> l) { 
  return ValueRange<T>(l); 
}

template <typename Range1, typename Range2>
Range2D<Range1, Range2> range(Range1 r1, Range2 r2) {
  return Range2D<Range1, Range2>(r1, r2);
}

template <typename Range1, typename Range2, typename Range3>
Range3D<Range1, Range2, Range3> range(Range1 r1, Range2 r2, Range3 r3) {
  return Range3D<Range1, Range2, Range3>(r1, r2, r3);
}

template <typename Range1, typename Range2, typename Range3, typename Range4>
Range4D<Range1, Range2, Range3, Range4> range(Range1 r1, Range2 r2, Range3 r3,
                                              Range4 r4) {
  return Range4D<Range1, Range2, Range3, Range4>(r1, r2, r3, r4);
}

#endif // include guard