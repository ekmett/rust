//! This module is a fork of the `bitvector_simd` crate, so I can use it with rank9 and add minor things, like len()
// iterators without fully forming the usizes into a vector, and u64 block retrieval. Several minor performance
// improvements have been made to allow short circuiting or avoiding calculating unnecessary counts. The endianness
// of the bits in each SIMD word has been reversed, conversion to Vec<usize> no longer requires fully
// flattening the entire structure and now exploits bit-test tricks. Modified to further use aligned storage through
// the allocator API

//! ### Usage
//!
//! ```rust
//! use bit::vector::BitVector;
//!
//! let _ = BitVector::ones(1_792); //create a set containing 0 ..= 1791
//! let mut bitvector = BitVector::ones(1_000); //create a set containing 0 ..= 999
//! bitvector.set(1_999, true); // add 1999 to the set, bitvector will be automatically expanded
//! bitvector.set(500, false); // delete 500 from the set
//! // now the set contains: 0 ..=499, 501..=1999
//! assert_eq!(bitvector.get(500), Some(false));
//! assert_eq!(bitvector.get(5_000), None);
//! // When try to get number larger than current bitvector, it will return `None`.
//! // of course if you don't care, you can just do:
//! assert_eq!(bitvector.get(5_000).unwrap_or(false), false);
//!
//! let bitvector2 = BitVector::zeros(2000); // create a set containing 0 ..=1999
//!
//! let bitvector3 = bitvector.and_cloned(&bitvector2);
//! // and/or/xor/not operation is provided.
//! // these APIs usually have 2 version:
//! // `.and` consume the inputs and `.and_clone()` accepts reference and will do clone on inputs.
//! let bitvector4 = bitvector & bitvector2;
//! // ofcourse you can just use bit-and operator on bitvectors, it will also consumes the inputs.
//! assert_eq!(bitvector3, bitvector4);
//! // A bitvector can also be constructed from a collection of bool, or a colluction of integer:
//! let bitvector: BitVector = (0 .. 10).map(|x| x%2 == 0).into();
//! let bitvector2: BitVector = (0 .. 10).map(|x| x%3 == 0).into();
//! let bitvector3 = BitVector::from_bool_iterator((0..10).map(|x| x%6 == 0));
//! assert_eq!(bitvector & bitvector2, bitvector3)
//! ```
//!
//! ## Performance
//!
//! run `cargo bench` to see the benchmarks on your device.
use std::{
  collections::TryReserveError,
  fmt::{self, Display},
  iter::{IntoIterator, Iterator},
  ops::{BitAnd, BitOr, BitXor, Index, Not},
  vec,
};

use crate::block::{self, *};
use crate::iter::set_bits;

//#[cfg(target_pointer_width = "32")]
//use packed_simd::u32x16;
//#[cfg(target_pointer_width = "32")]
//type Block = u32x16;

/// Representation of a BitVector
///
/// see the module's document for examples and details.
///
#[derive(Debug, Clone)]
pub struct BitVector {
  // internal representation of bitvector
  storage: Vec<Block>,
  // actual number of bits that exist in storage
  // invariant: x.storage.len() = (x.nbits + 511)/512
  nbits: usize,
}

/// Proc macro cannot export BitVector
/// macro_rules! cannot concat idents
/// so we use name, name_2 for function names
macro_rules! impl_operation {
  ($name:ident, $name_2:ident, $op:tt, $e:expr) => {
    #[must_use]
    pub fn $name(self, other: Self) -> Self {
      let nbits = self.nbits;
      assert_eq!(nbits, other.nbits);
      let mut storage = Vec::with_capacity(self.storage.len());
      self.storage
          .into_iter()
          .zip(other.storage.into_iter())
          .map(|(a, b)| a $op b)
          .collect_into(&mut storage);

      if ($e) {
        let r = nbits % 512;
        if r != 0 {
          storage[nbits / 512] &= block::mask(r);
        }
      }
      Self { storage, nbits }
    }
    #[must_use]
    pub fn $name_2(&self, other: &Self) -> Self {
      let nbits = self.nbits;
      assert_eq!(nbits, other.nbits);
      let mut storage = Vec::with_capacity(self.storage.len());
      self.storage
          .iter()
          .cloned()
          .zip(other.storage.iter().cloned())
          .map(|(a, b)| a $op b)
          .collect_into(&mut storage);
      if ($e) {
        let r = nbits % 512;
        if r != 0 {
          storage[nbits / 512] &= block::mask(r);
        }
      }
      Self {
        storage,
        nbits,
      }
    }
  };
}

impl BitVector {
  fn valid(&self) -> bool {
    let sq = (self.nbits + 511) / 512;
    self.storage.len() == sq && {
      let r = self.nbits % 512;
      r == 0 || (self.storage[sq - 1] & !block::mask(r) == block::ZERO)
    }
  }
  /// create an empty bit vector, but with a pre-allocated backing
  /// array large enough to store at least capacity-many bits.
  /// `capacity` will be rounded up to an integral multiple of
  /// `BLOCK_size`.
  pub fn with_capacity(capacity: usize) -> Self {
    let storage = Vec::with_capacity((capacity + 511) / 512);
    Self { storage, nbits: 0 }
  }
  /// Create a empty bitvector with `nbits` initial elements.
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::zeros(10);
  /// assert_eq!(bitvector.len(), 10);
  /// ```
  pub fn zeros(nbits: usize) -> Self {
    let sq = (nbits + 511) / 512;
    let storage = vec![block::ZERO;sq];
    Self { storage, nbits }
  }

  /// Create a bitvector containing all 0 .. nbits elements.
  pub fn ones(nbits: usize) -> Self {
    let (sq, r) = ((nbits + 511) / 512, nbits % 512);
    let mut storage = vec![block::ONES;sq];
    if r > 0 {
      storage[sq - 1] = block::mask(r)
    }
    Self { storage, nbits }
  }

  pub fn from_elem(elem: bool, nbits: usize) -> Self {
    if elem {
      Self::ones(nbits)
    } else {
      Self::zeros(nbits)
    }
  }

  /// Create a bitvector from an Iterator of bool.
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::from_bool_iterator((0..10).map(|x| x % 2 == 0));
  /// assert_eq!(bitvector.len(), 10);
  /// let actual = <BitVector as Into<Vec<bool>>>::into(bitvector);
  /// let golden = vec![true, false, true, false, true, false, true, false, true, false];
  /// assert_eq!(actual, golden, "actual\n{actual:?}\ngolden\n{golden:?}\n");
  ///
  /// let bitvector = BitVector::from_bool_iterator((0..1000).map(|x| x < 50));
  /// assert_eq!(bitvector.len(), 1000);
  /// assert_eq!(bitvector.get(49), Some(true));
  /// assert_eq!(bitvector.get(50), Some(false));
  /// assert_eq!(bitvector.get(999), Some(false));
  /// assert_eq!(<BitVector as Into<Vec<bool>>>::into(bitvector), (0..1000).map(|x| x<50).collect::<Vec<bool>>());
  /// ```
  pub fn from_bool_iterator<I: Iterator<Item = bool>>(i: I) -> Self {
    // FIXME: any better implementation?
    let lower_size_bound = i.size_hint().0;
    let mut storage = Vec::with_capacity((lower_size_bound + 511) / 512);
    let mut current_slice = [0u64; 8];
    let mut nbits = 0;
    for b in i {
      if b {
        current_slice[nbits % 512 / 64] |= 1 << ((nbits % 64) as u32);
      }
      nbits += 1;
      if nbits % 512 == 0 {
        storage.push(Block::from_slice_unaligned(&current_slice));
        current_slice = [0u64; 8];
      }
    }
    if nbits % 512 > 0 {
      storage.push(Block::from_slice_unaligned(&current_slice));
    }
    Self { storage, nbits }
  }

  /// Initialize from a set of integers.
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::from_slice(&[0,5,9]);
  /// let message = bitvector.clone();
  /// assert_eq!(<BitVector as Into<Vec<bool>>>::into(bitvector), vec![true, false, false, false, false, true, false, false, false, true],"into constructed the wrong bitvector: {message:?}");
  /// ```
  pub fn from_slice(slice: &[usize]) -> Self {
    let mut bv = BitVector::with_capacity(0);
    for i in slice {
      bv.set(*i, true);
    }
    bv
  }

  /// Max number of elements that this bitvector can have.
  ///
  /// To get the number of elements that are set, use `count`
  pub fn len(&self) -> usize {
    self.nbits
  }

  // how many elements this bitvector can grow to without reallocation
  // of the backing store.

  // NB: changed behavior relative to bitvector_simd
  pub fn capacity(&self) -> usize {
    self.storage.len() * BLOCK_SIZE
  }

  pub fn reserve(&mut self, additional: usize) {
    self.storage.reserve((self.nbits + additional + 511) / 512)
  }

  pub fn reserve_exact(&mut self, additional: usize) {
    self
      .storage
      .reserve_exact((self.nbits + additional + 511) / 512)
  }

  pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self
      .storage
      .try_reserve((self.nbits + additional + 511) / 512)
  }

  pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self
      .storage
      .try_reserve_exact((self.nbits + additional + 511) / 512)
  }

  pub fn clear(&mut self) {
    self.storage.clear();
    self.nbits = 0;
  }

  pub fn swap_remove(&mut self, index: usize) {
    #[cold]
    #[inline(never)]
    fn assert_failed(index: usize, len: usize) -> ! {
      panic!("swap_remove index (is {index}) should be < len (is {len})");
    }
    let len = self.nbits;
    if index >= len {
      assert_failed(index, len);
    }
    self.set(index, self.get(len - 1).unwrap());
    self.set(len - 1, false);
    self.nbits -= 1;
  }

  pub fn push(&mut self, value: bool) {
    self.set(self.nbits, value)
  }

  ///
  /// Shrink the vector to `length`. All elements between [length .. self.len()] will be removed.
  /// Panic if given `length` larger than current length.
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let mut bitvector = BitVector::ones(100);
  /// assert_eq!(bitvector.len(), 100);
  /// bitvector.truncate(10);
  /// assert_eq!(bitvector.len(), 10);
  /// // Now only contains [0 ..= 9]
  /// assert_eq!(bitvector.get(9), Some(true));
  /// assert_eq!(bitvector.get(10), None);
  /// ```
  ///
  /// NB: this was called `shrink_to` in `bitvector_simd`
  /// removed panic on under shrinkage to match vec api
  pub fn truncate(&mut self, length: usize) {
    if length < self.nbits {
      debug_assert!(self.valid());
      assert_eq!((self.nbits + 511) / 512, self.storage.len());
      let n = (length + 511) / 512;
      self.storage.truncate(n);
      self.nbits = length;
      let k = length % 512;
      if k > 0 {
        self.storage[n - 1] &= block::mask(k)
      }
      debug_assert!(self.valid());
    }
  }

  pub fn shrink_to(&mut self, min_capacity: usize) {
    self.storage.shrink_to((min_capacity + 511) / 512)
  }

  pub fn shrink_to_fit(&mut self) {
    self.storage.shrink_to_fit()
  }

  /// Remove or add `index` to the set.
  /// If index > self.len(), the bitvector will be expanded to `index`.
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let mut bitvector = BitVector::zeros(10);
  /// assert_eq!(bitvector.len(), 10);
  /// bitvector.set(15, true);  
  /// // now 15 has been added to the set, its total len is 16.
  /// assert_eq!(bitvector.len(), 16);
  /// assert_eq!(bitvector.get(15), Some(true));
  /// assert_eq!(bitvector.get(14), Some(false));
  /// ```
  pub fn set(&mut self, index: usize, flag: bool) {
    let nbits = self.nbits;
    if nbits <= index {
      let n = (index + 512) / 512 - (nbits + 511) / 512;
      self.storage.extend((0..n).map(move |_| block::ZERO));
      self.nbits = index + 1;
    }
    let (q, r) = (index / 512, index % 512);
    unsafe {
      let block = self.storage.get_unchecked_mut(q);
      *block = block::set_unchecked(*block, r, flag);
    };
  }

  /// Check if `index` exists in current set.
  ///
  /// * If exists, return `Some(true)`
  /// * If index < current.len and element doesn't exist, return `Some(false)`.
  /// * If index >= current.len, return `None`.
  ///
  /// Examlpe:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector : BitVector = (0 .. 15).map(|x| x%3 == 0).into();
  /// assert_eq!(bitvector.get(3), Some(true));
  /// assert_eq!(bitvector.get(5), Some(false));
  /// assert_eq!(bitvector.get(14), Some(false));
  /// assert_eq!(bitvector.get(15), None);
  /// ```
  pub fn get(&self, index: usize) -> Option<bool> {
    debug_assert!(self.valid(), "not valid {self}");
    let nbits = self.nbits;
    if nbits <= index {
      None
    } else {
      unsafe { Some(block::get_unchecked(self.storage[index / 512], index % 512)) }
    }
  }

  /// Directly return a `bool` instead of an `Option`
  ///
  /// * If exists, return `true`.
  /// * If doesn't exist, return false.
  /// * If index >= current.len, panic.
  ///
  ///
  /// Examlpe:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector : BitVector = (0 .. 15).map(|x| x%3 == 0).into();
  /// assert_eq!(bitvector.get_unchecked(3), true);
  /// assert_eq!(bitvector.get_unchecked(5), false);
  /// assert_eq!(bitvector.get_unchecked(14), false);
  /// ```
  pub fn get_unchecked(&self, index: usize) -> bool {
    if self.nbits <= index {
      panic!("index out of bounds {} > {}", index, self.nbits);
    } else {
      unsafe { block::get_unchecked(self.storage[index / 512], index % 512) }
    }
  }

  // todo: foo_borrowed?
  impl_operation!(and, and_cloned, &, false);
  impl_operation!(or, or_cloned, |, false);
  impl_operation!(xor, xor_cloned, ^, false);

  /// difference operation
  ///
  /// `A.difference(B)` calculates `A\B`, e.g.
  ///
  /// ```text
  /// A = [1,2,3], B = [2,4,5]
  /// A\B = [1,3]
  /// ```
  ///
  /// also notice that
  ///
  /// ```text
  /// A.difference(B) | B.difference(A) == A ^ B
  /// ```
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector: BitVector = (0 .. 5_000).map(|x| x % 2 == 0).into();
  /// let bitvector2 : BitVector = (0 .. 5_000).map(|x| x % 3 == 0).into();
  /// assert_eq!(bitvector.difference_borrowed(&bitvector2) | bitvector2.difference_borrowed(&bitvector), bitvector.xor_cloned(&bitvector2));
  /// let bitvector3 : BitVector = (0 .. 5_000).map(|x| x % 2 == 0 && x % 3 != 0).into();
  /// assert_eq!(bitvector.difference(bitvector2), bitvector3);
  /// ```

  /// TODO: use a simd andnot
  pub fn difference(self, other: Self) -> Self {
    let nbits = self.nbits;
    assert_eq!(nbits, other.nbits);
    let mut storage = Vec::with_capacity(self.storage.len());
    self
      .storage
      .into_iter()
      .zip(other.storage.into_iter())
      .map(|(a, b)| a & !b)
      .collect_into(&mut storage);

    let r = nbits % 512;
    if r != 0 {
      storage[nbits / 512] &= block::mask(r);
    }

    Self { storage, nbits }
  }

  pub fn difference_borrowed(&self, other: &Self) -> Self {
    let nbits = self.nbits;
    assert_eq!(nbits, other.nbits);
    let mut storage = Vec::with_capacity(self.storage.len());
    self
      .storage
      .iter()
      .zip(other.storage.iter())
      .map(|(a, b)| *a & !*b)
      .collect_into(&mut storage);

    let r = nbits % 512;
    if r != 0 {
      storage[nbits / 512] &= block::mask(r);
    }

    Self { storage, nbits }
  }

  // not should make sure bits > nbits is 0
  /// inverse every bits in the vector.
  ///
  /// If your bitvector have len `1_000` and contains `[1,5]`,
  /// after inverse it will contains `0, 2..=4, 6..=999`
  pub fn inverse(self) -> Self {
    let mut storage = Vec::with_capacity(self.storage.len());
    self
      .storage
      .into_iter()
      .map(|x| !x)
      .collect_into(&mut storage);
    let (q, r) = (self.nbits / 512, self.nbits % 512);
    storage[q] &= block::mask(r);
    Self {
      storage,
      nbits: self.nbits,
    }
  }

  /// Count the number of elements existing in this bitvector.
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector: BitVector = (0..10_000).map(|x| x%2==0).into();
  /// assert_eq!(bitvector.count(), 5000);
  ///
  /// let bitvector: BitVector = (0..30_000).map(|x| x%3==0).into();
  /// assert_eq!(bitvector.count(), 10_000);
  /// ```
  pub fn count(&self) -> usize {
    self
      .storage
      .iter()
      .map(|x| x.count_ones())
      .sum::<Block>()
      .wrapping_sum() as usize

    //.map(|x| x.count_ones().wrapping_sum())
    //.sum::<u64>() as usize
  }

  /// return true if contains at least 1 element
  pub fn any(&self) -> bool {
    self.storage.iter().any(move |x| *x != block::ZERO)
  }

  /// return true if contains self.len elements

  /// provides early exit relative to bitvector_simd impl
  pub fn all(&self) -> bool {
    debug_assert!(self.valid(), "not valid: {self}");
    let q = self.nbits / 512;
    self.storage.iter().take(q).all(move |x| *x == block::ONES) && {
      let r = self.nbits % 512;
      r == 0 || { self.storage[q] == block::mask(r) }
    }
  }

  /// return true if all bits in the bit vector are not set
  pub fn none(&self) -> bool {
    let z = Block::splat(0);
    self.storage.iter().all(move |x| *x == z)
  }

  /// Return true if bit vector has length 0.
  /// NB: changed behavior relative to bitvector_simd
  pub fn is_empty(&self) -> bool {
    self.nbits == 0
  }

  /// Consume self and generate a `Vec<bool>` with length == self.len().
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::from_bool_iterator((0..10).map(|i| i % 3 == 0));
  /// let bool_vec = bitvector.into_bools();
  /// assert_eq!(bool_vec, vec![true, false, false, true, false, false, true, false, false, true])
  /// ```
  pub fn into_bools(self) -> Vec<bool> {
    self.into()
  }

  /// Consume self and geterate a `Vec<usize>` which only contains the numbers that exist in this set.
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::from_bool_iterator((0..10).map(|i| i%3 == 0));
  /// let usize_vec = bitvector.into_usizes();
  /// assert_eq!(usize_vec, vec![0,3,6,9]);
  /// ```
  pub fn into_usizes(self) -> Vec<usize> {
    self.into()
  }

  /// Only compare the first `bits` instead of the whole bitvector.
  ///
  /// Require self and other are both no shorter than `bits`.
  ///
  /// Example:
  ///
  /// ```rust
  /// use bit::vector::BitVector;
  ///
  /// let bitvector = BitVector::from_slice(&[1,3,5,7]);
  /// let bitvector2 = BitVector::from_slice(&[1,3,5,9,10,15]);
  /// // compare first 6 bits (0..=5)
  /// assert!(bitvector.eq_left(&bitvector2, 6));
  /// // compare first 8 bits (0..=7)
  /// assert!(!bitvector.eq_left(&bitvector2, 8));
  /// // any bits > 8 call panic.
  /// ```
  pub fn eq_left(&self, other: &Self, bits: usize) -> bool {
    assert!(self.nbits >= bits && other.nbits >= bits);
    let (q, r) = (bits / 512, bits % 512);
    self
      .storage
      .iter()
      .zip(other.storage.iter())
      .take(q)
      .all(|(a, b)| a == b)
      && (r == 0 || {
        let m = block::mask(r);
        self.storage[q] & m == other.storage[q] & m
      })
  }
}

impl<I: Iterator<Item = bool>> From<I> for BitVector {
  fn from(i: I) -> Self {
    Self::from_bool_iterator(i)
  }
}

impl From<BitVector> for Vec<bool> {
  fn from(v: BitVector) -> Self {
    let mut result = Vec::with_capacity(v.nbits);
    v.storage
      .into_iter()
      .flat_map(|x| {
        let mut slice = [0u64; 8];
        // Packed SIMD does not provide any API to directly transform x into a slice
        // x.extract will consume itself which makes remaining data unaccessable.
        x.write_to_slice_unaligned(&mut slice);
        slice
      })
      .flat_map(|x| (0..u64::BITS).map(move |i| (x >> i) & 1 > 0))
      .take(v.nbits)
      .collect_into(&mut result);
    result
  }
}

impl IntoIterator for BitVector {
  type Item = usize;

  type IntoIter = impl Iterator<Item = usize>;

  fn into_iter(self) -> Self::IntoIter {
    let n = self.nbits;
    self
      .storage
      .into_iter()
      .flat_map(|x| {
        let mut slice = [0u64; 8];
        x.write_to_slice_unaligned(&mut slice);
        slice
      })
      .enumerate()
      .flat_map(|(i, y)| set_bits(y).map(move |k| i * 64 + k as usize))
      .filter(move |i| *i < n)
  }
}

impl From<BitVector> for Vec<usize> {
  fn from(v: BitVector) -> Self {
    v.into_iter().collect()
  }
}

impl Index<usize> for BitVector {
  type Output = bool;
  fn index(&self, index: usize) -> &Self::Output {
    if self.get_unchecked(index) {
      &true
    } else {
      &false
    }
  }
}

impl Display for BitVector {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let (q, r) = (self.nbits / 512, self.nbits % 512);
    for index in 0..q {
      let s = self.storage[index];
      for u in 0..8 {
        write!(f, "{:064b} ", s.extract(u))?;
      }
      writeln!(f, "")?;
    }
    if r > 0 {
      let s = self.storage[q];
      let qr = (r + 63) / 64;
      for u in 0..qr {
        write!(f, "{:064b} ", s.extract(u))?;
        writeln!(f, "")?;
      }
    }
    Ok(())
  }
}

impl PartialEq for BitVector {
  // eq should always ignore the bits > nbits
  fn eq(&self, other: &Self) -> bool {
    assert_eq!(self.nbits, other.nbits);
    self
      .storage
      .iter()
      .zip(other.storage.iter())
      .all(|(a, b)| a == b)
  }
}

impl BitAnd for BitVector {
  type Output = Self;
  fn bitand(self, rhs: Self) -> Self::Output {
    self.and(rhs)
  }
}

impl BitOr for BitVector {
  type Output = Self;
  fn bitor(self, rhs: Self) -> Self::Output {
    self.or(rhs)
  }
}

impl BitXor for BitVector {
  type Output = Self;
  fn bitxor(self, rhs: Self) -> Self::Output {
    self.xor(rhs)
  }
}

impl Not for BitVector {
  type Output = Self;
  fn not(self) -> Self::Output {
    self.inverse()
  }
}

pub trait BitStore {
  #[must_use]
  fn storage(&self) -> &Vec<Block>;
  #[must_use]
  fn len(&self) -> usize;
}

impl BitStore for BitVector {
  #[inline]
  fn storage(&self) -> &Vec<Block> {
      &self.storage
  }
  #[inline]
  fn len(&self) -> usize { self.nbits }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bit_vec_eqleft() {
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    assert!(bitvec.eq_left(&bitvec2, 1000));
    bitvec.set(900, false);
    assert!(bitvec.eq_left(&bitvec2, 900));
    assert!(bitvec.eq_left(&bitvec2, 800));
    assert!(bitvec.eq_left(&bitvec2, 900));
    assert!(!bitvec.eq_left(&bitvec2, 901));
    assert!(!bitvec.eq_left(&bitvec2, 1000));
  }

  #[test]
  fn test_bit_vec_count() {
    let mut bitvec = BitVector::ones(1000);
    assert_eq!(bitvec.count(), 1000, "{bitvec}");
    bitvec.set(1500, true);
    assert_eq!(bitvec.count(), 1001, "{bitvec}");
    bitvec.truncate(1);
    assert_eq!(bitvec.count(), 1, "{bitvec}");
  }

  #[test]
  fn test_bit_vec_all_any() {
    let trivial = BitVector::zeros(0);
    assert!(trivial.all(), "{trivial:?}");
    assert!(trivial.none(), "{trivial:?}");
    assert!(!trivial.any(), "{trivial:?}");
    let mut bitvec = BitVector::ones(1000);
    assert!(bitvec.all(), "{bitvec}");
    assert!(bitvec.any(), "{bitvec}");
    assert!(!bitvec.none(), "{bitvec}");
    bitvec.set(10, false);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    let mut bitvec = BitVector::zeros(1000);
    assert!(!bitvec.all());
    assert!(!bitvec.any());
    assert!(bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
  }

  #[test]
  fn test_bitvec_and_xor() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.xor_cloned(&bitvec2), BitVector::zeros(1000));
    assert_eq!(bitvec.xor_cloned(&bitvec3), BitVector::ones(1000));
    assert_eq!(bitvec ^ bitvec2, BitVector::zeros(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec ^ bitvec2;
    //println!("{}",bitvec3[400]);
    assert!(bitvec3[400], "{bitvec3}");
    assert_eq!(bitvec3.count(), 1);
  }

  #[test]
  fn test_bitvec_and_or() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.or_cloned(&bitvec2), BitVector::ones(1000));
    assert_eq!(bitvec.or_cloned(&bitvec3), BitVector::ones(1000));
    assert_eq!(bitvec | bitvec2, BitVector::ones(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec | bitvec2;
    assert!(bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count(), 1000);
  }

  #[test]
  fn test_bitvec_and_and() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.and_cloned(&bitvec2), BitVector::ones(1000));
    assert_eq!(bitvec.and_cloned(&bitvec3), BitVector::zeros(1000));
    assert_eq!(bitvec & bitvec2, BitVector::ones(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec & bitvec2;
    assert!(!bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count(), 1000 - 1);
  }

  #[test]
  fn test_bitvec_truncate() {
    let mut bitvec = BitVector::ones(1000);
    bitvec.truncate(900);
    assert_eq!(bitvec, BitVector::ones(900));
    bitvec.set(2000, true);
    assert!(bitvec.get_unchecked(2000));
    bitvec.truncate(1000);
    let mut bitvec2 = BitVector::ones(900);
    bitvec2.set(999, false);
    assert_eq!(bitvec, bitvec2);
  }

  #[test]
  fn test_bitvec_not() {
    let bitvec = BitVector::ones(1000);
    assert_eq!(bitvec, BitVector::ones(1000));
    assert_eq!(bitvec.not(), BitVector::zeros(1000));
  }

  #[test]
  fn test_bitvec_eq() {
    let mut bitvec = BitVector::ones(1000);
    assert_eq!(bitvec, BitVector::ones(1000));
    assert_ne!(bitvec, BitVector::zeros(1000));
    bitvec.set(50, false);
    assert_ne!(bitvec, BitVector::ones(1000));
    bitvec.set(50, true);
    assert_eq!(bitvec, BitVector::ones(1000));
  }

  #[test]
  fn test_bitvec_creation() {
    let mut bitvec = BitVector::zeros(1000);
    for i in 0..1500 {
      if i < 1000 {
        assert_eq!(bitvec.get(i), Some(false), "{bitvec} @ {i}");
      } else {
        assert_eq!(bitvec.get(i), None);
      }
    }
    bitvec.set(900, true);
    assert!(bitvec.valid(), "{bitvec}");
    for i in 0..1500 {
      if i < 1000 {
        assert_eq!(bitvec.get(i), Some(i == 900), "{i}:\n{bitvec}");
      } else {
        assert_eq!(bitvec.get(i), None, "{i}\n");
      }
    }
    bitvec.set(1300, true);
    for i in 0..1500 {
      if i <= 1300 {
        if i == 900 || i == 1300 {
          assert_eq!(bitvec.get(i), Some(true));
        } else {
          assert_eq!(bitvec.get(i), Some(false));
        }
      } else {
        assert_eq!(bitvec.get(i), None);
      }
    }
  }
}
