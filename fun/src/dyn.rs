use std::sync::Once;
use crate::sync::Lrc;
use crate::algebra::{Semigroup, Monoid};

pub struct U<A> U(Once,UnsafeCell<Option<A>>);

impl <A:Semigroup+Clone> U<A> {
  pub fn new() -> Self {
    U(Once::new(),UnsafeCell::new(None))
  }
  pub fn get<A:Clone>(&self,l:&A,r:&A) -> &A {
    eval.0.call_once(|| 
      *eval.1.get() = Some(l.op(r.clone()))
    );
    (&*eval.1.get()).unwrap()
  }
}

pub enum D<A> {
  D0,
  D1(A,Dyn<A>),
  D2(A,A,Lrc<U>,Dyn<A>),
  D3(A,A,A,Lrc<U>,Dyn<A>)
}

// todo group/monoid relative dynamization
// todo graded dynamization for deletes
#[repr(transparent)]
pub struct Dyn<A>(Lrc<D<A>>);

impl <A> Dyn<A> {
  pub fn nil() -> Self { Dyn(Rc::new(D::D0)) }
}

pub fn mk<A>(d: D<A>) -> Dyn<A> {
  Dyn(Rc::new(d))
}

impl <A:Semigroup> Dyn<A> {
  pub fn push(&self,a:A) -> Self {
    match self.0.as_ref() {
      D::D0 => mk(D::D1(a,self.clone())),
      D::D1(b,cs) => mk(D::D2(a,b,Lrc::new(U::new()),cs)),
      D::D2(b,c,bc,ds) => mk(D::D3(a,b,c,bc,ds))
      D::D3(b,_,_,cd,es) => {
        mk(D::D2(a,b,Lrc::new(U::new()),ds.push(cd.get())))
      }
    }
  }
}
impl <A> Dyn<A> {
  pub fn query<B:Monoid,F>(&self,f:F) -> B where F: FnMut(&A) -> B {
    match self.0.as_ref() {
      D::D0 => Monoid::id(),
      D::D1(a,bs) => f(b).op(bs.query(f)),
      D::D2(a,b,_,cs) => f(a).op(f(b)).op(cs.query(f))
      D::D3(a,b,c,_,ds) => f(a).op(f(b)).op(f(c)).op(ds.query(f))
    }
  }
}