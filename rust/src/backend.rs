use anyhow::Result;

use crate::{Basis, Bitmap, ResultSet, Vector, ID};

#[allow(clippy::module_name_repetitions)]
pub struct BackendInfo {
    pub quantization: String,
    pub has_index_data: bool,
    pub dimensions: usize,
    pub n_basis: usize,
    pub vector_count: usize,
}

#[allow(clippy::missing_errors_doc, clippy::module_name_repetitions)]
pub trait VectorBackend {
    fn put_vector(&mut self, id: ID, v: &Vector) -> Result<()>;
    fn compute_similarity(&self, target: &Vector, target_id: ID) -> Result<f32>;
    fn info(&self) -> BackendInfo;
    fn iter_vector_ids(&self) -> impl Iterator<Item = ID>;
    fn vector_exists(&self, id: ID) -> bool;
    fn close(self) -> Result<()>;

    /// # Errors
    /// `compute_similarity` can fail
    fn find_nearest(&self, target: &Vector, k: usize) -> Result<ResultSet> {
        let mut set = ResultSet::new(k);
        for id in self.iter_vector_ids() {
            let sim = self.compute_similarity(target, id)?;
            set.add_result(id, sim);
        }
        Ok(set)
    }

    fn load_bases(&self) -> Result<Option<Vec<Basis>>>;
    fn load_bitmap<B: Bitmap>(&mut self, basis: usize, index: i32) -> Result<Option<B>>;

    /// # Errors
    /// none
    fn save_bases(&mut self, _bases: &[Basis]) -> Result<()> {
        Ok(())
    }

    /// # Errors
    /// none
    fn save_bitmap(&mut self, _basis: usize, _index: usize, _bitmap: &impl Bitmap) -> Result<()> {
        Ok(())
    }

    /// # Errors
    /// none
    fn sync(&self) -> Result<()> {
        Ok(())
    }
}
