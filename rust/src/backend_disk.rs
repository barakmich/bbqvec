use anyhow::Result;
use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    quantization::Quantization, vector_file::VectorFile, Basis, Bitmap, Vector, VectorBackend, ID,
};

#[derive(Default)]
pub struct DiskBackend<Q: Quantization> {
    dir: PathBuf,
    metadata: DiskMetadata,
    vector_files: HashMap<usize, VectorFile<Q>>,
    token: u64,
}

#[derive(Serialize, Deserialize, Default)]
pub(crate) struct DiskMetadata {
    pub dimensions: usize,
    pub quantization: String,
    pub vecs_per_file: usize,
    pub vec_files: Vec<usize>,
}

const DEFAULT_VECS_PER_FILE: usize = 200_000;

impl<Q: Quantization> DiskBackend<Q> {
    pub fn open(path: PathBuf, dimensions: usize) -> Result<Self> {
        let mut token: u64 = rand::random();
        if token == 0 {
            token = 1;
        }
        let mut s = Self {
            dir: path,
            metadata: DiskMetadata {
                dimensions,
                quantization: Q::name().into(),
                vecs_per_file: DEFAULT_VECS_PER_FILE,
                vec_files: Vec::new(),
            },
            token,
            ..Default::default()
        };
        s.open_files()?;
        Ok(s)
    }

    fn open_files(&mut self) -> Result<()> {
        let metadata_path = self.dir.join("metadata.json");
        if !metadata_path.exists() {
            return self.create_new();
        }
        let metadata_contents = std::fs::read_to_string(&metadata_path)?;
        let metadata: DiskMetadata = serde_json::from_str(&metadata_contents)?;
        self.metadata = metadata;
        for vf in self.metadata.vec_files.iter() {
            let vector_file = VectorFile::<Q>::create_or_open(
                self.make_pagefile_path(vf),
                self.metadata.dimensions,
                self.metadata.vecs_per_file,
            )?;
            self.vector_files.insert(*vf, vector_file);
        }
        Ok(())
    }

    fn create_new(&mut self) -> Result<()> {
        std::fs::create_dir_all(self.dir.clone())?;
        self.save_metadata()
    }

    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.dir.join("metadata.json");
        Ok(serde_json::to_writer(
            &std::fs::File::create(metadata_path)?,
            &self.metadata,
        )?)
    }

    fn make_pagefile_path(&self, key: &usize) -> PathBuf {
        self.dir.join(format!("{:x}.vec", key))
    }
}

impl<Q: Quantization> VectorBackend for DiskBackend<Q> {
    fn put_vector(&mut self, id: ID, v: &Vector) -> Result<()> {
        todo!()
    }

    fn compute_similarity(&self, target: &Vector, target_id: ID) -> Result<f32> {
        todo!()
    }

    fn info(&self) -> crate::backend::BackendInfo {
        todo!()
    }

    fn iter_vector_ids(&self) -> impl Iterator<Item = ID> {
        0..(self.metadata.vecs_per_file * self.metadata.vec_files.len()) as ID
    }

    fn vector_exists(&self, id: ID) -> bool {
        todo!()
    }

    fn close(self) -> Result<()> {
        todo!()
    }

    fn load_bases(&self) -> Result<Option<Vec<Basis>>> {
        todo!()
    }

    fn load_bitmap<B: Bitmap>(&mut self, basis: usize, index: i32) -> Result<Option<B>> {
        todo!()
    }

    fn save_bases(&mut self, _bases: &[Basis]) -> Result<()> {
        Ok(())
    }

    fn save_bitmap(&mut self, _basis: usize, _index: usize, _bitmap: &impl Bitmap) -> Result<()> {
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        for v in self.vector_files.values() {
            v.flush()?
        }
        self.save_metadata()
    }
}
