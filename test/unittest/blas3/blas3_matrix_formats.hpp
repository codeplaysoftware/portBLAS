#pragma once

// define a simple structure for representing (at the type level) various formats of matrix transposition

class Normal {
  public:
  static constexpr char const* str = "n";
};

class Transposed {
  public:
  static constexpr char const* str = "t";
};

class Conjugate {
  public:
   static constexpr char const* str = "c";
};

template <class AT_ = Normal, class BT_ = Normal>
struct MatrixFormats
{
  using a_format = AT_;
  using b_format = BT_;
};