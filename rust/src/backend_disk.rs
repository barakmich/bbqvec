use anyhow::Result;
use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{quantization::Quantization, vector_file::VectorFile};

#[derive(Default)]
pub struct DiskBackend<Q: Quantization> {
    dir: PathBuf,
    metadata: DiskMetadata,
    vector_files: HashMap<i64, VectorFile<Q>>,
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
        todo!()
    }
}
