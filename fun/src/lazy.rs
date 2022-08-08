#![feature(trusted_len)]
#![feature(exact_size_is_empty)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![allow(dead_code)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(trivial_numeric_casts)]

use std::borrow::Borrow;
use std::boxed::Box;
use std::cell::UnsafeCell;
use std::fmt::{self, Debug, Formatter};
use std::mem;
use std::ops::{Deref, Fn, FnMut, FnOnce};
use crate::sync::Lrc;

// almost has the semantics of a scala lazy val
// but requires mutation. attempts to generally
// follow the poison semantics of mutexes and 
// Once blocks.
pub enum Closure<'f, T> {
  Delayed(Box<dyn (FnOnce() -> T) + 'f>),
  Forced(T),
  Poisoned
}

impl<'f, T: Debug> Debug for Closure<'f, T> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Closure::Delayed(_) => f.write_str("<closure>"),
      Closure::Forced(t) => Debug::fmt(&t, f),
      Closure::Poisoned => f.write_str("<poisoned>")
    }
  }
}

impl<'f, T: 'f> Closure<'f, T> {
  #[inline]
  pub fn new<F: (FnOnce() -> T) + 'f>(f: F) -> Self {
    Closure::Delayed(Box::new(f))
  }

  #[inline]
  pub const fn ready(&self) -> bool {
    matches!(self,Closure::Forced(_))
  }

  #[inline]
  pub const fn is_poisoned(&self) -> bool {
    matches!(self,Closure::Poisoned)
  }

  #[inline]
  pub fn seq(&mut self) {
    let mf = if let Closure::Delayed(ref mut fr) = self {
      Some(mem::replace(fr, detail::blackhole()))
    } else {
      None
    };
    if let Some(f) = mf {
      let result = panic::catch_unwind(|| {
        f()
      });
      match result {
      Ok(t) => *self = Closure::Forced(t),
      Err(e) => {
        *self = Closure::Poisoned;
        panic::resume_unwind(err);
      }
    }
  }
  #[inline]
  pub fn get(&mut self) -> &T {
    self.seq();
    match self {
      Closure::Forced(ref t) => &t,
      Closure::Poisoned => panic!("poisoned"),
      Closure::Delayed(_) => unreachable!()
    }
  }

  #[inline]
  pub const fn try_get(&self) -> Option<&T> {
    if let Closure::Forced(ref t) = self {
      Some(&t)
    } else {
      None
    }
  }
  #[inline]
  pub fn consume(self) -> T {
    match self {
      Closure::Delayed(f) => f(),
      Closure::Forced(t) => t,
      Closure::Poisoned => panic!("poisoned")
    }
  }

  #[inline]
  pub fn try_consume(self) -> Result<T,Self> {
    match self {
      Closure::Forced(t) => Ok(t),
      _ => Err(self),
    }
  }
  #[inline]
  pub fn map_consume<'g, U, F>(self, f: F) -> Closure<'g, U>
  where
    'f: 'g,
    U: 'g,
    T: 'g,
    F: (FnOnce(&T) -> U) + 'g, {
      Closure::new(move || f(&self.consume()))
  }

  pub fn promote(&mut self) -> Lazy<'f, T>
  where T: 'f + Clone, {
    if let Closure::Forced(value) = self {
      Lazy::from(value.clone())
    } else {
      let placeholder = Closure::new(|| unreachable!());
      let old_guts = mem::replace(self, placeholder);
      let result = Lazy(Lrc::new(LazyVal(UnsafeCell::new(old_guts))));
      let clone = result.clone();
      let new_guts = Closure::new(move || clone.get().clone());
      let _ = mem::replace(self, new_guts);
      result
    }
  }
}

impl<'f, T: 'f> From<T> for Closure<'f, T> {
  #[inline]
  fn from(that: T) -> Self {
    Closure::Forced(that)
  }
}

impl<'f, T: 'f + Default> Default for Closure<'f, T> {
  fn default() -> Self {
    Closure::new(|| T::default())
  }
}

impl<'f, T: 'f> IntoIterator for Closure<'f, T> {
  type Item = T;
  type IntoIter = detail::ClosureIterator<'f, T>;
  fn into_iter(self) -> Self::IntoIter {
    detail::ClosureIterator(Some(self))
  }
}

// this is a scala-style 'lazy val'. with all the upsides
// and downsides that would entail
#[derive(Debug)]
pub struct LazyVal<'f, T>(UnsafeCell<Closure<'f, T>>);

impl<'f, T: 'f> LazyVal<'f, T> {
  #[inline]
  pub fn new<F: (FnOnce() -> T) + 'f>(f: F) -> Self {
    LazyVal(UnsafeCell::new(Closure::new(f)))
  }
  #[inline]
  pub fn seq(&self) {
    unsafe { &mut *self.0.get() }.seq()
  }
  #[inline]
  pub fn ready(&self) -> bool {
    unsafe { &*self.0.get() }.ready()
  }
  #[inline]
  pub fn get<'t,'u>(&'t self) -> &'u T where 't: 'u, 'f:'u {
    unsafe { &mut *self.0.get() }.get()
  }
  #[inline]
  pub const fn is_poisoned(&self) -> bool {
    unsafe { &*self.0.get() }.is_poisoned()
  }
  #[inline]
  pub fn try_get(&self) -> Option<&'f T> {
    unsafe { &*self.0.get() }.try_get()
  }
  #[inline]
  pub fn consume(self) -> T {
    self.0.into_inner().consume()
  }
  #[inline]
  pub fn try_consume(self) -> Result<T,Self> {
    self.0.into_inner().try_consume().map_err(|e| LazyVal(UnsafeCell::new(e)))
  }
  #[inline]
  pub fn map_consume<'g, U, F>(self, f: F) -> LazyVal<'g, U> where
    'f: 'g, U: 'g, T: 'g, F: (FnOnce(&'f T) -> U) + 'g, {
      LazyVal::new(move || f(self.get()))
  }
  #[inline]
  pub fn promote(&self) -> Lazy<'f, T> where
    T: 'f + Clone, {
      unsafe { &mut *self.0.get() }.promote()
  }
}

impl<'f, T: Default + 'f> Default for LazyVal<'f, T> {
  fn default() -> Self {
    LazyVal::new(|| T::default())
  }
}

impl<'f, T: 'f> From<Closure<'f, T>> for LazyVal<'f, T> {
  fn from(that: Closure<'f, T>) -> Self {
    LazyVal(UnsafeCell::new(that))
  }
}

impl<'f, T: 'f> From<T> for LazyVal<'f, T> {
  fn from(that: T) -> Self {
    LazyVal::from(Closure::from(that))
  }
}

impl<'f, T: 'f> From<LazyVal<'f, T>> for Closure<'f, T> {
  fn from(that: LazyVal<'f, T>) -> Self {
    that.0.into_inner()
  }
}

impl<'f, T: 'f> Borrow<T> for LazyVal<'f, T> {
  fn borrow(&self) -> &T {
    self.get()
  }
}

impl<'f, T: 'f> AsRef<T> for LazyVal<'f, T> {
  fn as_ref(&self) -> &T {
    self.get()
  }
}

impl<'f, T: 'f> Deref for LazyVal<'f, T> {
  type Target = T;
  fn deref(&self) -> &T {
    self.get()
  }
}

impl<'f, T: 'f> IntoIterator for LazyVal<'f, T> {
  type Item = T;
  type IntoIter = detail::ClosureIterator<'f, T>;
  fn into_iter(self) -> Self::IntoIter {
    self.0.into_inner().into_iter()
  }
}

// a haskell-style thunk, single threaded
#[derive(Debug)]
#[repr(transparent)]
pub struct Lazy<'f, T: 'f>(pub Rc<LazyVal<'f, T>>);

impl<'f, T: 'f> Clone for Lazy<'f, T> {
  fn clone(&self) -> Self {
    Lazy(self.0.clone())
  }

  fn clone_from(&mut self, source: &Self) {
    self.0.clone_from(&source.0)
  }
}

impl<'f, T: 'f> Lazy<'f, T> {
  pub fn new<F: (FnOnce() -> T) + 'f>(f: F) -> Self {
    Lazy(Lrc::new(LazyVal::new(f)))
  }
  pub fn new_strict(value: T) -> Self {
    Lazy(Lrc::new(LazyVal::from(value)))
  }
  pub fn seq(&self) {
    self.0.as_ref().seq()
  }
  pub fn ready(&self) -> bool {
    self.0.as_ref().ready()
  }
  pub fn get(&self) -> &'f T {
    self.0.as_ref().get()
  }
  #[inline]
  pub const fn is_poisoned(&self) -> bool {
    self.0.as_ref().is_poisoned()
  }
  pub fn try_get(&self) -> Option<&'f T> {
    self.0.as_ref().try_get()
  }
  pub fn map<'g, U, F: (FnOnce(&T) -> U) + 'g>(&self, f: F) -> Lazy<'g, U> where
    'f: 'g, T: 'g, U: 'g, {
    let me = self.clone();
    Lazy::new(move || f(me.get()))
  }
  pub fn map2<'g, 'h, U, V, F: (FnOnce(&T, &U) -> V) + 'h>(
    this: &Lazy<'f, T>,
    that: &Lazy<'g, U>,
    f: F,
  ) -> Lazy<'h, V> where
    'f: 'h, 'g: 'h, T: 'h, U: 'h, {
    let a = this.0.clone();
    let b = that.0.clone();
    Lazy::new(move || f(a.get(), b.get()))
  }

  // consumes this lazy value in an effort to try to avoid cloning the contents
  pub fn consume(self) -> T where T: Clone, {
    match Lrc::try_unwrap(self.0) {
      Result::Ok(lval) => lval.consume(),
      Result::Err(this) => this.get().clone(), // other references to this thunk exist
    }
  }
  pub fn try_consume(self) -> Result<T,Self> where T: Clone, {
    match Lrc::try_unwrap(self.0) {
      Result::Ok(lval) => lval.try_consume().map_err(|e| Lazy(Rc::new(e))),
      Result::Err(this) => match this.try_get() {
        Some(x) => Ok(x.clone()),
        None => Result::Err(Lazy(this))
      },
    }
  }
}

impl<'f, T: 'f + Default> Default for Lazy<'f, T> {
  fn default() -> Self {
    Lazy::new(|| T::default())
  }
}

impl<'f, T: 'f> From<T> for Lazy<'f, T> {
  #[inline]
  fn from(that: T) -> Self {
    Lazy::new_strict(that)
  }
}

impl<'f, T: 'f> FnOnce<()> for Lazy<'f, T> {
  type Output = &'f T;
  extern "rust-call" fn call_once(self, _args: ()) -> &'f T {
    self.0.as_ref().get()
  }
}

impl<'f, T: 'f> FnMut<()> for Lazy<'f, T> {
  extern "rust-call" fn call_mut(&mut self, _args: ()) -> &'f T {
    self.0.as_ref().get()
  }
}

impl<'f, T: 'f> Fn<()> for Lazy<'f, T> {
  extern "rust-call" fn call(&self, _args: ()) -> &'f T {
    self.0.as_ref().get()
  }
}

impl<'f, T: 'f> Borrow<T> for Lazy<'f, T> {
  fn borrow(&self) -> &T {
    self.get()
  }
}

impl<'f, T: 'f> AsRef<T> for Lazy<'f, T> {
  fn as_ref(&self) -> &T {
    self.get()
  }
}

// impl<'f, T: 'f> Deref for Lazy<'f, T> {
//   type Target = T;
//   fn deref(&self) -> &T {
//     self.get()
//   }
// }

impl<'f, T: 'f + Clone> IntoIterator for Lazy<'f, T> {
  type Item = T;
  type IntoIter = detail::LazyIterator<'f, T>;
  fn into_iter(self) -> Self::IntoIter {
    detail::LazyIterator(Some(self))
  }
}

/*
impl <'f, T : Clone> Future for Lazy<'f,T> {
  type Output = T;
  fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
    let this = self.into_inner()
    if Some(t) = self.try_get()
    ...
  }
}
*/

pub mod detail {
  use std::iter::{ExactSizeIterator,TrustedLen,FusedIterator};
  use super::*;

  pub fn blackhole<'f, T: 'f>() -> Box<dyn (FnOnce() -> T) + 'f> {
    Box::new(|| panic!("<infinite loop>"))
  }

  pub fn promoting<'f, T: 'f>() -> Box<dyn (FnOnce() -> T) + 'f> {
    Box::new(|| unreachable!())
  }

  #[derive(Debug)]
  #[repr(transparent)]
  pub struct ClosureIterator<'f, T: 'f>(pub Option<Closure<'f, T>>);

  impl<'f, T: 'f> Iterator for ClosureIterator<'f, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
      Some(self.0.take()?.consume())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
      let n = self.len();
      (n, Some(n))
    }
    fn last(self) -> Option<Self::Item> {
      Some(self.0?.consume())
    }
    fn count(self) -> usize {
      if self.0.is_some() {
          1
      } else {
          0
      }
    }
  }

  impl <'f,T> FusedIterator for ClosureIterator<'f,T> {}
  unsafe impl<'f,T: 'f> TrustedLen for ClosureIterator<'f, T> {}
  impl <'f,T: 'f> ExactSizeIterator for ClosureIterator<'f, T> {
    fn len(&self) -> usize {
      if self.0.is_some() { 1 } else { 0 }
    }

    fn is_empty(&self) -> bool {
      self.0.is_none()
    }
  }

  #[derive(Debug)]
  #[repr(transparent)]
  pub struct LazyIterator<'f, T: 'f>(pub Option<Lazy<'f, T>>);

  impl<'f, T: 'f> Clone for LazyIterator<'f, T> {
    #[inline]
    fn clone(&self) -> Self {
      LazyIterator(self.0.clone())
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
      self.0.clone_from(&source.0)
    }
  }
  impl<'f, T: 'f + Clone> Iterator for LazyIterator<'f, T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
      Some(self.0.take()?.consume())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
      let n = if self.0.is_some() { 1 } else { 0 };
      (n, Some(n))
    }
    fn last(self) -> Option<Self::Item> {
      Some(self.0?.consume())
    }
    fn count(self) -> usize {
      if self.0.is_some() {
        1
      } else {
        0
      }
    }
  }

  impl <'f,T: 'f + Clone> FusedIterator for LazyIterator<'f,T> {}
  unsafe impl<'f,T: 'f + Clone> TrustedLen for LazyIterator<'f, T> {}
  impl <'f,T: 'f + Clone> ExactSizeIterator for LazyIterator<'f, T> {
    fn len(&self) -> usize {
      if self.0.is_some() { 1 } else { 0 }
    }

    fn is_empty(&self) -> bool {
      self.0.is_none()
    }
  }

}

pub fn main() {
  let mut y = 12;
  println!("{}", y);
  let x = Lazy::new(|| {
    println!("x forced");
    y += 1;
    y * 10
  });
  let w = x.map(|r| r + 1);
  println!("{}", w());
  println!("{}", w.get());
  // println!("{}", *w); // deref makes for nice syntax
  for z in w {
    println!("{}", z);
  }
}
