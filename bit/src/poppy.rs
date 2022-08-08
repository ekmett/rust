
use crate::block::{self,Block};
use std::ops::{BitAnd, BitOr, BitXor};
use crate::vector::BitStore;
use std::vec::Vec;
use std::fmt;

type L0 = usize;
pub const L0_STRIDE : usize = 4194304; // Blocks

#[derive(Debug,Copy,Clone)]
pub struct L12(u32,u32);
type SuperBlock = [Block;4];
pub const L12_STRIDE : usize = 4; // 4 Blocks = 1 SuperBlock

// this could be made 2x faster with intrinsics
// but we don't seem to have good access to permutes?
// maybe use masking instead?
#[must_use]
fn scan_superblock(x: &mut u32, y: SuperBlock) -> L12 {
   let z : [u32;4] = [
       y[0].count_ones().wrapping_sum() as u32,
       y[1].count_ones().wrapping_sum() as u32,
       y[2].count_ones().wrapping_sum() as u32,
       y[3].count_ones().wrapping_sum() as u32
   ];
   let result = L12(*x,z[0]+(z[1]<<10)+(z[2]<<20));
   *x += z[0]+z[1]+z[2]+z[3];
   result
}




#[derive(Debug,Clone)]
pub struct Poppy<Store> {
  pub level0:  Vec<L0>,
  pub level12: Vec<L12>,
  pub store:   Store,
}

impl <Store:BitStore> BitStore for Poppy<Store> {
  #[inline]
  fn storage(&self) -> &Vec<Block> {
    self.store.storage()
  }

  #[inline]
  fn len(&self) -> usize {
    self.store.len()
  }
}

// TODO: allow for append-only growth
impl <Store:BitStore> Poppy<Store> {
  #[must_use]
  pub fn new(store: Store) -> Self {
    // now we need to construct the index.
    let nbits = store.len();
    let storage = store.storage();

    let mut level0_sum = 0usize;
    let level0_capacity = (nbits / (L0_STRIDE * 512)).saturating_sub(1);

    let mut level0 : Vec<usize> = Vec::with_capacity(level0_capacity);
    let mut level12 = Vec::with_capacity(nbits / (L12_STRIDE * 512) + 1);

    // avoid storing the first entry in the level0 cache
    let mut storing0 = false;
    // small enough to ensure that we don't overflow u32
    for level0_block in storage.chunks(L0_STRIDE) {
        if storing0 {
          level0.push(level0_sum);
        } else {
          storing0 = true;
        }
        let mut level12_sum = 0u32;
        let (whole, parts) = level0_block.as_chunks::<L12_STRIDE>();
        for superblock in whole {
            level12.push(scan_superblock(&mut level12_sum, *superblock));
        }
        // now we have to cobble together some kind of leftovers
        let k = parts.len();
        if k > 0 {
            let mut last : SuperBlock = [block::ZERO;4];
            for i in 0..k {
                last[i] = parts[i];
            }
            level12.push(scan_superblock(&mut level12_sum, last));
        }
        level0_sum += usize::try_from(level12_sum).unwrap();
        //level0.push(level0_sum);
    }
    //level12.push(L12(last_level12_sum,0));
    Poppy { level0, level12, store }
  }

  #[inline]
  #[must_use]
  pub fn rank1(&self, mut index: usize) -> usize {
    index = index.max(self.len());
    if index == 0 { return 0 }
    index -= 1;

    let level0_count : usize = {
      let level0_index = index >> 31;
      if level0_index > 0 {
        self.level0[level0_index-1]
      } else {
        0
      }
    };

    // 0 case where we don't need the index is already handled above
    let L12(level1_count,level2_u32) = self.level12[index >> 11];

    let subblock = index >> 9;
    let m = level2_u32 & ((1 << (10 * (subblock & 3))) - 1);
    let level2_count = (m & 1023) + ((m >> 10) & 1023) + ((m >> 20) & 1023);

    let r = (index + 1) % 512;

    let level3_count =
      if r > 0 {
        let block = self.storage()[subblock];
        let masked_block = block & block::mask(index % 512);
        block::count_ones(masked_block)
      } else {
        0
      };

    let level123_count = usize::try_from(
      level1_count + level2_count + level3_count
    ).unwrap();

    level0_count + level123_count
  }

  #[inline]
  #[must_use]
  pub fn rank0(&self, index: usize) -> usize {
    index - self.rank1(index)
  }

}

impl <Store:IntoIterator> IntoIterator for Poppy<Store> {
    type Item = <Store as IntoIterator>::Item;
    type IntoIter = <Store as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}

impl <Store> From<Poppy<Store>> for Vec<bool> where
  Vec<bool> : From<Store>,
{
    fn from(it: Poppy<Store>) -> Self {
        <Vec<bool> as From<Store>>::from(it.store)
    }
}

impl<I, Store> From<I> for Poppy<Store> where
  I: Iterator<Item = bool>,
  Store : From<I> + BitStore,
{
    fn from(i: I) -> Self {
        Poppy::new(<Store as From<I>>::from(i))
    }
}

impl <Store: fmt::Display> fmt::Display for Poppy<Store> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.store.fmt(f)
    }
}

impl <Store: PartialEq> PartialEq for Poppy<Store> {
    fn eq(&self, other: &Self) -> bool {
        self.store == other.store
    }
}

impl <Store: Eq> Eq for Poppy<Store> {}

impl <Store> BitAnd for Poppy<Store> where
  Store: BitStore + BitAnd<Store,Output=Store>,
{
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(self.store.bitand(rhs.store))
    }
}

impl <Store> BitOr for Poppy<Store> where
  Store: BitStore + BitOr<Store,Output=Store>,
{
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(self.store.bitor(rhs.store))
    }
}

impl <Store> BitXor for Poppy<Store> where
  Store: BitStore + BitXor<Store,Output=Store>,
{
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::new(self.store.bitxor(rhs.store))
    }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::vector::BitVector;
  #[test]
  fn it_works() {
    let p = Poppy::new(BitVector::from_slice(&[3,4,100]));
    assert_eq!(p.len(),101);
    assert_eq!(p.rank1(101),3);
    assert_eq!(p.rank1(100),2);
    assert_eq!(p.rank1(0),0);

    let q = Poppy::new(BitVector::from_slice(&[]));
    assert_eq!(q.len(),0);
    assert_eq!(q.rank1(0),0);
    assert_eq!(q.rank1(1),0);

  }

  fn test_ones(nbits: usize) {
    let r = Poppy::new(BitVector::ones(nbits));
    assert_eq!(r.rank1(nbits),nbits);
  }

  #[test]
  fn test_1024() {
    test_ones(1024);
  }

  #[test]
  fn test_small() {
    for i in 0..4096 {
      test_ones(i);
    }
  }
  #[test]
  fn test_huge() {
    test_ones(2usize <<32);
  }

}
