#![feature(
  const_trait_impl,
  is_sorted,
  trusted_len,
  exact_size_is_empty,
  iter_collect_into,
  type_alias_impl_trait,
  slice_as_chunks
)]

pub mod block;
pub mod iter;
pub mod poppy;
pub mod traits;
pub mod vector;
