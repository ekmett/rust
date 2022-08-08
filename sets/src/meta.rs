use std::convert::TryFrom;
use std::fmt::{self, Debug};
use std::num::{NonZeroUsize, TryFromIntError};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
#[repr(transparent)]
pub struct Meta(NonZeroUsize);

impl Meta {
  #[inline]
  pub fn new(i: usize) -> Option<Meta> {
    Some(Meta(NonZeroUsize::new(!i)?))
  }

  #[inline]
  pub unsafe fn new_unchecked(i: usize) -> Self {
    Meta(NonZeroUsize::new_unchecked(!i))
  }

  #[inline]
  pub fn usize(self) -> usize {
    !self.0.get()
  }

  #[inline]
  pub fn from_usize(i: usize) -> Result<Meta, TryFromIntError> {
    Ok(Meta(NonZeroUsize::try_from(!i)?))
  }
}

impl From<Meta> for usize {
  #[inline]
  fn from(id: Meta) -> usize {
    id.usize()
  }
}

impl TryFrom<usize> for Meta {
  type Error = TryFromIntError;
  #[inline]
  fn try_from(u: usize) -> Result<Meta, TryFromIntError> {
    Meta::from_usize(u)
  }
}

impl Default for Meta {
  #[inline]
  fn default() -> Self {
    unsafe { Self::new_unchecked(0) }
  }
}

impl fmt::Debug for Meta {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    Debug::fmt(&self.usize(), f)
  }
}
