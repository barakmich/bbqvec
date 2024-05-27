use std::ops::{BitAndAssign, BitOrAssign};

use backend::{BuildableBackend, VectorBackend};
use roaring::RoaringBitmap;

pub mod backend;
pub(crate) mod backend_memory;
pub(crate) mod counting_bitmap;
pub mod result;
pub(crate) mod spaces;
pub(crate) mod unaligned_f32;
pub(crate) mod vector;

pub use backend_memory::MemoryBackend;

use anyhow::Result;

pub use result::ResultSet;

pub type Vector = Vec<f32>;
pub type ID = u64;
pub type Basis = Vec<Vector>;

pub trait Bitmap: BitAndAssign + BitOrAssign + Default + Clone + serde::Serialize {
    fn new() -> Self;
    fn count(&self) -> usize;
    fn add(&mut self, id: ID);
    fn iter(&self) -> impl Iterator<Item = usize>;
    fn and(&mut self, rhs: &Self);
    fn or(&mut self, rhs: &Self);
}

impl Bitmap for roaring::RoaringBitmap {
    fn new() -> Self {
        RoaringBitmap::new()
    }

    fn count(&self) -> usize {
        self.len() as usize
    }

    fn add(&mut self, id: ID) {
        self.insert(id as u32);
    }

    fn iter(&self) -> impl Iterator<Item = usize> {
        self.iter().map(|x| x as usize)
    }
    fn and(&mut self, rhs: &Self) {
        self.bitand_assign(rhs)
    }
    fn or(&mut self, rhs: &Self) {
        self.bitor_assign(rhs)
    }
}

use bitvec::prelude::BitVec;

impl Bitmap for BitVec {
    fn new() -> Self {
        BitVec::new()
    }

    fn count(&self) -> usize {
        self.len()
    }

    fn add(&mut self, id: ID) {
        if self.len() < id as usize {
            self.resize(id as usize, false);
        }
        self.set(id as usize, true)
    }

    fn iter(&self) -> impl Iterator<Item = usize> {
        self.iter_ones()
    }
    fn and(&mut self, rhs: &Self) {
        self.bitand_assign(rhs)
    }

    fn or(&mut self, rhs: &Self) {
        self.bitor_assign(rhs)
    }
}

pub fn full_table_scan_search<B: BuildableBackend + VectorBackend>(
    backend: B,
    target: Vector,
    k: usize,
) -> Result<ResultSet> {
    let mut set = ResultSet::new(k);
    for (id, _) in backend.iter() {
        let sim = backend.compute_similarity(&target, id)?;
        set.add_result(id, sim);
    }
    Ok(set)
}
