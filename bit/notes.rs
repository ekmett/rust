// to perform a catamorphism, construct a sparse-dense array?
// alternatively, construct a bitmask, and use poppy and rank


// use slotmap::Key;
// use anymap;

// pub enum Strategy {
//   New,
//   WithCapacity(usize)
// }

// pub struct Constable<S = RandomState> {
//   strategy : Strategy,
//   slotmaps : AnyMap, // contains Slotmap<FooKey,Foo>
//   hashmaps : AnyMap, // hashmap HashMap<Foo,
//   marker   : PhantomData<* const S>
// }

// // associated with the keys
// trait HashKey : slotmap::Key where Self::Of : HasKey<Key=Self>, { type Of }

// // associated with the types
// trait HasKey where Self:Key : HashKey<Of=Self>, { type Key }

// type HM<K,S = RandomState> = HashMap<K,<V as HasKey>::Key,S>

// impl Constable<S = RandomState> {
//   fn new() -> Self { 
//     Constable { 
//       strategy: Strategy::New,
//       slotmaps: AnyMap::new(),
//       hashmaps: AnyMap::new(),
//       marker: PhantomData
//     }
//   }

//   fn with_capacity(am_capacity: usize, sm_capacity: usize) -> Self { 
//     Constable { 
//       strategy: Strategy::WithCapacity(sm_capacity),
//       slotmaps: AnyMap::with_capacity(am_capacity),
//       hashmaps: AnyMap::with_capacity(am_capacity), 
//       marker: PhantomData
//     }
//   }
//   // fn slotmap_for_key<K:HashKey>(...)I

//   fn get<K:HashKey>(&self) -> &<K as HashKey>::Of;

//   // todo steal the attribute tricks that makes specs fast? or just move to specs
//   fn get_hashmap<V:HasKey>(&self) -> HashMap<<V as HasKey>::Key,V,S> { self.hashmaps.get() }
//   fn get_slotmap<K:HashKey>(&self) -> SlotMap<K,<V as HashKey>::Of,S> { self.slotmaps.get() }
//   fn intern<V:HasKey>(&mut self, v:V) -> <V as HasKey>::Key {
//      if (let Some(vp) = self.hashmaps.get(v)) {
//        v
//      } 
//   }
//   fn unsafe_delete<K:HashKey>(&mut self,k:K) {

//   }
// }
