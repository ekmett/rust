use packed_simd::{u64x8, m64x8, FromCast};

pub(crate) const WORD_SIZE : usize = 64;
pub(crate) const BLOCK_SIZE : usize = WORD_SIZE*8;

pub type Block = u64x8;
pub type Mask = m64x8;

pub const ZERO : Block = Block::splat(0u64);
pub const ONES : Block = Block::splat(u64::MAX);
//const ONE  : Block = Block::splat(1u64);
const I    : Block = Block::new(0,1,2,3,4,5,6,7);

// excessively clever manual simd-ification
#[must_use]
pub(crate) fn mask(k: usize) -> Block {
  debug_assert!(k <= 512);
  let (q,r) = (k / 64, k as u32 % 64);
  let maxes = Block::from_cast(I.lt(Block::splat(q as u64)));
  if r != 0 {
    unsafe { maxes.replace_unchecked(q,1u64.wrapping_shl(r).wrapping_sub(1u64)) }
  } else {
    maxes
  }
}

#[cfg(test)]
fn mask_simple(k: usize) -> Block {
  let mut slice = [0u64;8];
  let (q,r) = (k / 64, k as u32 % 64);
  for i in 0..q {
      slice[i] = u64::MAX;
  }
  if r != 0 {
      slice[q] = 1u64.wrapping_shl(r).wrapping_sub(1u64);
  }
  Block::from_slice_unaligned(&slice)
}  

#[test]
fn simd_mask_is_valid() {
  for i in 0..512 {
    assert_eq!(mask(i),mask_simple(i),"failed to match at index {i}");
  }
}


#[inline]
#[must_use]
#[allow(dead_code)]
pub(crate) fn unpack(bc: Block) -> [u64;8] {
    let mut slice = [0u64;8];
    unsafe { bc.write_to_slice_unaligned_unchecked (&mut slice); }
    slice
}

#[inline]
#[must_use]
#[allow(dead_code)]
pub(crate) fn pack(slice: [u64;8]) -> Block {
    unsafe { Block::from_slice_unaligned_unchecked(&slice) }
}

#[inline]
#[must_use]
#[allow(dead_code)]
pub(crate) fn count_ones(bc: Block) -> u32 {
    Block::count_ones(bc).wrapping_sum() as u32
}

#[allow(dead_code)]
pub(crate) unsafe fn get_unchecked(bc: Block, i: usize) -> bool {
    let (q,r) = (i / 64, i % 64);
    (Block::extract_unchecked(bc,q) >> r) & 1 != 0
}

#[inline]
pub(crate) unsafe fn set_unchecked(bc: Block, i: usize, bit: bool) -> Block {
  let (q,r) = (i/64,i%64);
  let s = bc.extract(q);
  let sp = if bit {
    s | (1u64 << r)
  } else {
    s & !(1u64 << r)
  };
  bc.replace_unchecked(q,sp)
}

#[test]
fn tests() {
    // assert_eq!(mask(1))
    assert_eq!(mask(0),pack([0u64;8]));
    assert_eq!(mask(65),pack([u64::MAX,1,0,0,0,0,0,0]));
    assert_eq!(mask(511),pack([u64::MAX,u64::MAX,u64::MAX,u64::MAX,u64::MAX,u64::MAX,u64::MAX,u64::MAX>>1]));
    assert_eq!(mask(512),Block::splat(u64::MAX));
    assert_eq!(mask(1),mask(3) & mask(1));
    for i in 0..512 {
        assert_eq!(count_ones(mask(i as usize)),i, "{i}");
    }

    unsafe {
      for i in 0..512 {
          for j in 0..512 {
              assert_eq!(get_unchecked(mask(i),j),i > j,"failed at {i}, {j}")
          }
      }
    }
}

