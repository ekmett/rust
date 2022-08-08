use std::ops::{BitAnd, BitXorAssign, Shl};
use num_traits::{WrappingNeg, identities::{Zero,One}};

pub trait CountOnes {
  type Output;
  fn count_ones(self) -> Self::Output;
}
pub trait CountZeros {
  type Output;
  fn count_zeros(self) -> Self::Output;
}
pub trait LeadingOnes {
  type Output;
  fn leading_ones(self) -> Self::Output;
}
pub trait LeadingZeros {
  type Output;
  fn leading_zeros(self) -> Self::Output;
}
pub trait TrailingOnes {
  type Output;
  fn trailing_ones(self) -> Self::Output;
}
pub trait TrailingZeros {
  type Output;
  fn trailing_zeros(self) -> Self::Output;
}
pub trait HasBits {
  type Output;
  #[allow(non_snake_case)]
  fn BITS() -> Self::Output;
}

// a poor approximation of the Haskell Bits class
// means "these work nicely together" in a sort of DWIM way
pub trait Bits
      : TrailingZeros<Output = u32>
      + LeadingZeros<Output = u32>
      + TrailingZeros<Output = u32>
      + CountOnes<Output = u32>
      + WrappingNeg
      + BitAnd<Output = Self>
      + PartialEq
      + Shl<u32, Output = Self>
      + Zero
      + One
      + Copy
      + HasBits<Output = u32>
      + BitXorAssign<Self> {}

impl <T> Bits for T where
    T : TrailingZeros<Output = u32>
      + LeadingZeros<Output = u32>
      + CountOnes<Output = u32>
      + WrappingNeg
      + BitAnd<Output = T>
      + PartialEq
      + Shl<u32, Output = T>
      + Zero
      + One
      + Copy
      + HasBits<Output = u32>
      + BitXorAssign<Self> {}

pub unsafe trait TrustedBits : Bits {}

macro_rules! bit_impl {
  ($($t:ty)*) => ($(
    impl const CountOnes for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn count_ones(self) -> Self::Output { <$t>::count_ones(self) }
    }
    impl const CountZeros for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn count_zeros(self) -> Self::Output { <$t>::count_zeros(self) }
    }
    impl const LeadingOnes for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn leading_ones(self) -> Self::Output { <$t>::leading_ones(self) }
    }
    impl const LeadingZeros for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn leading_zeros(self) -> Self::Output { <$t>::leading_zeros(self) }
    }
    impl const TrailingOnes for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn trailing_ones(self) -> Self::Output { <$t>::trailing_ones(self) }
    }
    impl const TrailingZeros for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      fn trailing_zeros(self) -> Self::Output { <$t>::trailing_zeros(self) }
    }
    impl const HasBits for $t {
      type Output = u32;
      #[inline]
      #[must_use]
      #[allow(non_snake_case)]
      fn BITS() -> Self::Output { <$t>::BITS }
    }
    unsafe impl TrustedBits for $t {}
  )*)
}

bit_impl!(u8 u16 u32 u64 usize i8 i16 i32 i64 isize);
