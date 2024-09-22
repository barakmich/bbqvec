use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::path::PathBuf;

use crate::quantization::Quantization;

pub struct VectorFile<Q: Quantization> {
    dimensions: usize,
    vec_size: usize,
    mmap: MmapMut,
    max_vecs: usize,
    quantization: std::marker::PhantomData<Q>,
}

impl<Q: Quantization> VectorFile<Q> {
    pub fn create_or_open(path: PathBuf, dimensions: usize, max_vecs: usize) -> Result<Self> {
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) => panic!("Failed to open or create file: {}", e),
        };
        let vec_size = Q::vector_size(dimensions);

        let file_size = max_vecs * vec_size;
        if file.metadata().unwrap().len() == 0 {
            file.set_len(file_size as u64).unwrap();
            file.sync_data()?;
        }
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            dimensions,
            vec_size,
            mmap,
            max_vecs,
            quantization: Default::default(),
        })
    }

    pub fn flush(&self) -> Result<()> {
        Ok(self.mmap.flush_async()?)
    }

    pub fn write_at(&mut self, offset: usize, vec: &Q::Lower) -> Result<()> {
        if offset >= self.max_vecs {
            return Err(anyhow!("Offset outside file bounds"));
        }
        let slice = &mut self.mmap[offset * self.vec_size..(offset + 1) * self.vec_size];
        Q::marshal(vec, slice)
    }

    pub fn read_at(&self, offset: usize) -> Result<Q::Lower> {
        if offset >= self.max_vecs {
            return Err(anyhow!("Offset outside file bounds"));
        }
        let slice = &self.mmap[offset * self.vec_size..(offset + 1) * self.vec_size];
        Q::unmarshal(slice)
    }
}
