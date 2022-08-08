#![allow(dead_code)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(trivial_numeric_casts)]

pub mod meta;

use std::mem::swap;

use meta::Meta;

// Using Rem's algorithm rather than the standard Tarjan tricks.
// https://drops.dagstuhl.de/opus/volltexte/2020/11801/pdf/LIPIcs-OPODIS-2019-15.pdf
//
// rem's algorithm has the benefit of becoming more stable over time as existing 
// Metas can only increase, and because they are complemented, this points down towards
// the bottom of the disjoint_sets structure.

#[derive(Debug, Clone, Default)]
pub struct Sets(Vec<Meta>);

impl Sets {
  pub fn with_capacity(capacity: usize) -> Self {
    Sets(Vec::with_capacity(capacity))
  }

  pub fn new() -> Self {
    Sets(Vec::new())
  }

  pub fn make_set(&mut self) -> Meta {
    let id = unsafe { Meta::new_unchecked(self.len()) };
    self.0.push(id);
    id
  }

  pub fn len(&self) -> usize { self.0.len() }

  pub fn capacity(&self) -> usize { self.0.capacity() }

  pub fn parent(&self, p: Meta) -> Meta { self.0[p.usize()] }

  fn parent_mut(&mut self, p: Meta) -> &mut Meta { &mut self.0[p.usize()] }

  // find without self-modification
  pub fn find(&self, mut p: Meta) -> Meta {
    while p != self.parent(p) {
      p = self.parent(p);
    }
    p
  }

  pub fn find_mut(&mut self, mut p: Meta) -> Meta {
    while p != self.parent(p) {
      let gp = self.parent(self.parent(p));
      *self.parent_mut(p) = gp;
      p = gp
    }
    p
  }

  // make them equal and returns the first node at which this becomes true
  pub fn union(&mut self, mut u: Meta, mut v: Meta) -> Meta {
    loop {
      let mut up = self.parent(u);
      let mut vp = self.parent(v);
      if u == v || up == vp { 
        break up 
      }
      if vp < up {
        swap(&mut u,&mut v);
        swap(&mut up,&mut vp);
      }
      if u == up {
        // if we're doing this multithreaded then
        // if compare_and_swap(self.parent_mut(u),u,vp) { return vp; } 
        *self.parent_mut(u) = vp;
        break vp;
      }
      v = self.parent(up);
      if up != v {
        // if we're doing this multithreaded then
        // compare_and_swap(self.parent_mut(u), up, v)
        *self.parent_mut(u) = v;
      }
      u = up;
    }
  }

  // match the behavior of a more traditional union_find
  pub fn union_find(&mut self, u: Meta, v: Meta) -> Meta {
    let w = self.union(u,v);
    self.find_mut(w)
  }

  pub fn same(&mut self, mut u: Meta, mut v: Meta) -> bool {
    loop { 
      let mut up = self.parent(u);
      let mut vp = self.parent(v);
      if u == v || up == vp {
        break true
      }
      if vp < up { 
        swap(&mut u,&mut v);
        swap(&mut up,&mut vp);
      }
      if u == up { 
        break false
      }
      v = self.parent(up);
      if up != v {
        // compare_and_swap(self.parent_mut(u), up, v)
        *self.parent_mut(u) = v;
      }
      u = up;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn it_works() {
    let mut ds = Sets::with_capacity(2);
    assert_eq!(ds.capacity(),2);
    assert_eq!(ds.len(), 0);
    let x = ds.make_set();
    let y = ds.make_set();
    let z = ds.make_set();
    assert!(ds.same(x,x));
    assert!(ds.same(y,y));
    assert!(!ds.same(x,y));
    ds.union(x,y);
    assert!(ds.same(x,y));
    assert!(ds.same(x,x));
    assert!(ds.same(y,y));
    assert!(!ds.same(x,z));
    ds.union(x,z);
    assert!(ds.same(y,z));
    assert_eq!(ds.len(), 3);
    let u = ds.make_set();
    let v = ds.make_set();
    let w = ds.union_find(u,v);
    let vr = ds.find_mut(v);
    assert_eq!(w,vr); // known to be roots
  }
}
