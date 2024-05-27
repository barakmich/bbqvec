use anyhow::Result;

use crate::{Basis, Bitmap, Vector, ID};

pub struct BackendInfo {
    pub has_index_data: bool,
    pub dimensions: usize,
    pub n_basis: usize,
    pub vector_count: usize,
}

pub trait VectorBackend: Sized {
    type Buildable: BuildableBackend;
    type Indexable: IndexableBackend;
    type Compiling: CompilingBackend;

    fn put_vector(&mut self, id: ID, v: &Vector) -> Result<()>;
    fn compute_similarity(&self, target: &Vector, target_id: ID) -> Result<f32>;
    fn info(&self) -> BackendInfo;
    fn as_buildable_backend(&mut self) -> Option<&mut Self::Buildable>;
    fn as_indexable_backend<T: Bitmap>(&mut self) -> Option<&mut Self::Indexable>;
    fn as_compiling_backend(&mut self) -> Option<&mut Self::Compiling>;
}

pub trait BuildableBackend {
    fn get_vector(&self, id: ID) -> Result<Vector>;
    fn get_random_vector<R: rand::Rng>(&self, rng: R) -> Result<Vector>;
    fn iter(&self) -> impl Iterator<Item = (ID, &Vector)>;
}

pub trait IndexableBackend {
    fn save_bases(&mut self, bases: &[Basis]) -> Result<()>;
    fn load_bases(&self) -> Result<Vec<Basis>>;
    fn save_bitmap(&mut self, basis: usize, index: usize, bitmap: &impl Bitmap) -> Result<()>;
    fn load_bitmap<T: Bitmap>(&mut self, basis: usize, index: usize) -> Result<T>;
}

pub trait CompilingBackend {
    fn compile(&mut self) -> Result<()>;
}
