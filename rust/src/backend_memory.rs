use std::{
    cmp::min,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use rand::RngCore;

use crate::{
    backend::{BackendInfo, VectorBackend},
    quantization::Quantization,
    Vector, ID,
};

pub struct QuantizedMemoryBackend<Q: Quantization> {
    vecs: Vec<Option<Q::Lower>>,
    dimensions: usize,
    n_basis: usize,
    rng: Option<Arc<Mutex<Box<dyn RngCore + Send>>>>,
}

pub type MemoryBackend = QuantizedMemoryBackend<crate::quantization::NoQuantization>;

impl<Q: Quantization> QuantizedMemoryBackend<Q> {
    pub fn new(dimensions: usize, n_basis: usize) -> Result<Self> {
        Ok(Self {
            vecs: Vec::new(),
            dimensions,
            n_basis,
            rng: None,
        })
    }

    pub fn set_rng(&mut self, rng: Box<dyn RngCore + Send>) {
        self.rng = Some(Arc::new(Mutex::new(rng)));
    }
}

impl<Q: Quantization> VectorBackend for QuantizedMemoryBackend<Q> {
    fn put_vector(&mut self, id: crate::ID, v: &Vector) -> Result<()> {
        if v.len() != self.dimensions {
            return Err(anyhow!("dimensions don't match"));
        }
        let uid = id as usize;
        if self.vecs.len() <= uid {
            if self.vecs.capacity() == uid {
                self.vecs.reserve(min(self.vecs.capacity(), 1024 * 1024))
            }
            self.vecs.resize(uid + 1, None);
        }
        let mut insert = v.clone();
        crate::vector::normalize(&mut insert);
        let l = Q::lower(insert)?;
        self.vecs[uid] = Some(l);
        Ok(())
    }

    fn compute_similarity(&self, target: &Vector, target_id: crate::ID) -> Result<f32> {
        // Make sure it's normalized!
        let v = self.vecs[target_id as usize]
            .as_ref()
            .ok_or(anyhow!("No vector present"))?;
        Q::compare(target, v)
    }

    fn info(&self) -> crate::backend::BackendInfo {
        BackendInfo {
            has_index_data: false,
            dimensions: self.dimensions,
            n_basis: self.n_basis,
            vector_count: self.vecs.len(),
            quantization: Q::name().into(),
        }
    }

    fn iter_vector_ids(&self) -> impl Iterator<Item = ID> {
        self.vecs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(k, _)| k as ID)
    }

    fn vector_exists(&self, id: ID) -> bool {
        let v = self.vecs.get(id as usize);
        match v {
            Some(x) => x.is_some(),
            None => false,
        }
    }

    fn close(self) -> Result<()> {
        todo!()
    }

    fn load_bases(&self) -> Result<Option<Vec<crate::Basis>>> {
        Ok(None)
    }

    fn load_bitmap<T: crate::Bitmap>(&mut self, _basis: usize, _index: i32) -> Result<Option<T>> {
        Ok(None)
    }
}
