use crate::traits::*;
use std::iter::{ExactSizeIterator,FusedIterator,DoubleEndedIterator,TrustedLen};

// loop over the set bits in a u64, etc.
// biterator?
pub struct Biterator<T>(T);

impl <T:Bits> Iterator for Biterator<T> {
  type Item = <T as TrailingZeros>::Output;
  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    if self.0 != T::zero() {
      let t = self.0 & self.0.wrapping_neg();
      self.0 ^= t;
      Some(t.trailing_zeros())
    } else {
      None
    }
  }

  #[inline]
  fn size_hint(&self) -> (usize, Option<usize>) {
    if let Ok(n) = usize::try_from(self.0.count_ones()) {
      (n,Some(n))
    } else {
      (0,None)
    }
  }

  #[inline]
  fn count(self) -> usize where Self: Sized, {
    usize::try_from(self.0.count_ones()).unwrap()
  }

  #[inline]
  fn last(self) -> Option<Self::Item> where Self: Sized, {
    if self.0 != T::zero() {
      Some(T::BITS() - self.0.leading_zeros() - 1)
    } else {
      None
    }
  }

  #[inline]
  fn max(self) -> Option<Self::Item> {
    self.last()
  }

  #[inline]
  fn min(self) -> Option<Self::Item> {
    if self.0 != T::zero() {
      Some((self.0 & self.0.wrapping_neg()).trailing_zeros())
    } else {
      None
    }
  }

  #[inline]
  fn is_sorted(self) -> bool { true }
}

impl <T:Bits> FusedIterator for Biterator<T> where {}

unsafe impl <T:TrustedBits> TrustedLen for Biterator<T> {}
impl <T:Bits> ExactSizeIterator for Biterator<T> where {
  #[inline]
  fn len(&self) -> usize {
    usize::try_from(self.0.count_ones()).unwrap()
  }

  #[inline]
  fn is_empty(&self) -> bool {
    self.0 == T::zero()
  }
}

impl <T:Bits> DoubleEndedIterator for Biterator<T> {
  #[inline]
  fn next_back(&mut self) -> Option<Self::Item> {
    if self.0 != T::zero() {
      let r = T::BITS() - self.0.leading_zeros() - 1;
      self.0 ^= T::one() << r;
      Some(r)
    } else {
      None
    }
  }
}

#[inline]
#[must_use]
pub fn set_bits<T:Bits + std::fmt::Debug>(t: T) -> Biterator<T> {
  println!("Construction set_bits: {t:?}");
  Biterator(t)
} 

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn it_works() {
      assert_eq!(set_bits(5).last(),Some(2u32));
      assert!(set_bits(9).eq([0u32,3u32]));
      assert!(set_bits(9).rev().eq([3u32,0u32]));
    }
}
