use std::cmp::min;

use anyhow::{anyhow, Result};

use crate::{
    backend::{BackendInfo, BuildableBackend, CompilingBackend, IndexableBackend, VectorBackend},
    Bitmap, Vector, ID,
};

pub struct MemoryBackend {
    vecs: Vec<Option<Vector>>,
    dimensions: usize,
    n_basis: usize,
}

impl MemoryBackend {
    pub fn new(dimensions: usize, n_basis: usize) -> Result<Self> {
        Ok(Self {
            vecs: Vec::new(),
            dimensions,
            n_basis,
        })
    }
}

impl VectorBackend for MemoryBackend {
    type Buildable = Self;
    type Indexable = Self;
    // When ! comes out of nightly, that's what I want here
    type Compiling = Self;

    fn put_vector(&mut self, id: crate::ID, v: &Vector) -> Result<()> {
        if v.len() != self.dimensions {
            return Err(anyhow!("dimensions don't match"));
        }
        let uid = id as usize;
        if self.vecs.len() < uid {
            if self.vecs.capacity() == uid {
                self.vecs.reserve(min(self.vecs.capacity(), 1024 * 1024))
            }
            self.vecs.resize(uid + 1, None);
        }
        let mut insert = v.clone();
        crate::vector::normalize(&mut insert);
        self.vecs[uid] = Some(insert);
        Ok(())
    }

    fn compute_similarity(&self, target: &Vector, target_id: crate::ID) -> Result<f32> {
        // Make sure it's normalized!
        let v = self.vecs[target_id as usize].as_ref().unwrap();
        Ok(crate::vector::dot_product(target, v))
    }

    fn info(&self) -> crate::backend::BackendInfo {
        BackendInfo {
            has_index_data: false,
            dimensions: self.dimensions,
            n_basis: self.n_basis,
            vector_count: self.vecs.len(),
        }
    }

    fn as_buildable_backend(&mut self) -> Option<&mut Self::Buildable> {
        Some(self)
    }

    fn as_indexable_backend<T: crate::Bitmap>(&mut self) -> Option<&mut Self::Indexable> {
        Some(self)
    }

    fn as_compiling_backend(&mut self) -> Option<&mut Self::Compiling> {
        None
    }
}

impl BuildableBackend for MemoryBackend {
    fn get_vector(&self, id: crate::ID) -> Result<Vector> {
        let elem = self.vecs.get(id as usize).ok_or(anyhow!("out of bounds"))?;
        elem.clone().ok_or(anyhow!("vector not found"))
    }

    fn get_random_vector<R: rand::Rng>(&self, mut rng: R) -> Result<Vector> {
        // This assumes a rather dense vector set... otherwise, use an ID lookup map
        let x: usize = rng.gen_range(0..self.vecs.len());
        self.get_vector(x as ID)
            .or_else(|_| self.get_random_vector(rng))
    }

    fn iter(&self) -> impl Iterator<Item = (ID, &Vector)> {
        self.vecs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(k, v)| (k as ID, v.as_ref().unwrap()))
    }
}

impl IndexableBackend for MemoryBackend {
    fn save_bases(&mut self, _bases: &[crate::Basis]) -> Result<()> {
        todo!()
    }

    fn load_bases(&self) -> Result<Vec<crate::Basis>> {
        todo!()
    }

    fn save_bitmap(&mut self, _basis: usize, _index: usize, _bitmap: &impl Bitmap) -> Result<()> {
        todo!()
    }

    fn load_bitmap<T: Bitmap>(&mut self, _basis: usize, _index: usize) -> Result<T> {
        todo!()
    }
}

impl CompilingBackend for MemoryBackend {
    fn compile(&mut self) -> Result<()> {
        panic!("unreachable")
    }
}
