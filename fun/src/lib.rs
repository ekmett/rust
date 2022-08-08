#![feature(strict_provenance)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(const_trait_impl)]
#![feature(associated_type_defaults)]
#![feature(never_type)]
#![feature(extend_one)]
#![feature(exact_size_is_empty)]
#![feature(trusted_len)]
// #![feature(fused)]
#![feature(arc_unwrap_or_clone)]


// === Linter configuration
#![allow(dead_code)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(trivial_numeric_casts)]

pub mod sync;
pub mod list;
pub mod skew;
pub mod queue;
pub mod cat;
pub mod algebra;
pub mod group_relative;
pub mod monoid_relative;
