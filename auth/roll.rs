
// can we build a rolling hash based skew tree?

// treap-list, store in, height based on hash

// adapton style lists, using treaps rather than skiplists
// TODO: replace Rc with Hc

#[repr(transparent)]
pub struct TList<A>(Option<Rc<(TList<A>,A,TList<A>)>>)

fn bin(l:TList<A>,a:A,r:TList<A>) -> TList<A> {
  TList::new(Some(Rc::new((l,a,r))))
}

pub const fn nil() -> TList<A> { TList(None) }

// O(log n) in expectation
pub fn cons<A:Hash+Clone>(a:A,t: &TList<A>) -> TList<A> {
  match t.as_ref() {
    None => bin(nil(),a,nil())
    Some((l,b,r)) => 
      if a.hash < b.hash {
        bin(nil(),a,t.clone())
      } else {
        bin(cons(a,l.clone()),b.clone(),r.clone())
      }
  }
}

// O(log n) in expectation
pub fn snoc<A:Hash+Clone>(t: &TList<A>,b:A) -> TList<A> {
  match t.as_ref() {
    None => bin(nil(),b,nil())
    Some((l,a,r)) => 
      if a.hash < b.hash {
        bin(t.clone(),a,nil())
      } else {
        bin(l.clone(),a.clone(),snoc(r.clone,b))
      }
  }
}

// O(log n) in expectation
pub fn link<A:Hash>(x: &TList<A>, y: &TList<A>) -> TList<A> {
  match x.as_ref() {
  None => y.clone(),
  Some((la,a,ra)) => 
    match y.as_ref() {
    None => x.clone(),
    Some((lb,b,rb)) => 
      if a.hash < b.hash {
        bin(la.clone(),a,link(ra,y))
      } else {
        bin(link(x,lb),b,rb.clone())
      }
    }
  }
}
