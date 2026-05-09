use crate::ID;
use std::ops::{BitOrAssign, BitXorAssign, SubAssign};

pub use bitvec::prelude::BitVec;
pub use croaring::Bitmap as CRoaringBitmap;
pub use roaring::RoaringBitmap;

pub trait Bitmap: std::fmt::Debug + Default + Clone + Send {
    fn new() -> Self;
    fn count(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn add(&mut self, id: ID);
    fn iter_elems(&self) -> impl Iterator<Item = ID>;
    fn and_not(&mut self, rhs: &Self);
    fn or(&mut self, rhs: &Self);
    fn xor(&mut self, rhs: &Self);
    fn estimate_size(&self) -> usize;
}

impl Bitmap for roaring::RoaringBitmap {
    fn new() -> Self {
        roaring::RoaringBitmap::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn count(&self) -> usize {
        self.len() as usize
    }

    fn add(&mut self, id: ID) {
        #[allow(clippy::cast_possible_truncation)]
        self.insert(id as u32);
    }

    fn iter_elems(&self) -> impl Iterator<Item = ID> {
        #[allow(clippy::cast_lossless)]
        self.iter().map(|x| x as ID)
    }
    fn and_not(&mut self, rhs: &Self) {
        self.sub_assign(rhs);
    }
    fn or(&mut self, rhs: &Self) {
        self.bitor_assign(rhs);
    }
    fn xor(&mut self, rhs: &Self) {
        self.bitxor_assign(rhs);
    }
    fn estimate_size(&self) -> usize {
        self.serialized_size()
    }
}

impl Bitmap for bitvec::prelude::BitVec {
    fn new() -> Self {
        bitvec::prelude::BitVec::new()
    }

    fn count(&self) -> usize {
        self.count_ones()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn add(&mut self, id: ID) {
        if self.len() <= id as usize {
            self.resize((id + 1) as usize, false);
        }
        self.set(id as usize, true);
    }

    fn iter_elems(&self) -> impl Iterator<Item = ID> {
        self.iter_ones().map(|x| x as ID)
    }

    #[inline]
    fn and_not(&mut self, rhs: &Self) {
        for elem in self.as_raw_mut_slice().iter_mut().zip(rhs.as_raw_slice()) {
            *elem.0 &= !elem.1;
        }
    }

    #[inline]
    fn or(&mut self, rhs: &Self) {
        if self.len() < rhs.len() {
            self.resize(rhs.len(), false);
        }
        self.bitor_assign(rhs);
    }

    #[inline]
    fn xor(&mut self, rhs: &Self) {
        if self.len() < rhs.len() {
            self.resize(rhs.len(), false);
        }
        self.bitxor_assign(rhs);
    }

    fn estimate_size(&self) -> usize {
        std::mem::size_of_val(self.as_raw_slice())
    }
}

impl Bitmap for croaring::Bitmap {
    fn new() -> Self {
        croaring::Bitmap::new()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn count(&self) -> usize {
        self.cardinality() as usize
    }

    fn add(&mut self, id: ID) {
        #[allow(clippy::cast_possible_truncation)]
        self.add(id as u32);
    }

    fn iter_elems(&self) -> impl Iterator<Item = ID> {
        #[allow(clippy::cast_lossless)]
        self.iter().map(|x| x as ID)
    }

    fn and_not(&mut self, rhs: &Self) {
        self.andnot_inplace(rhs);
    }

    fn or(&mut self, rhs: &Self) {
        self.or_inplace(rhs);
    }

    fn xor(&mut self, rhs: &Self) {
        self.xor_inplace(rhs);
    }

    fn estimate_size(&self) -> usize {
        self.get_serialized_size_in_bytes::<croaring::Native>()
    }
}
