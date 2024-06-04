use backend::BuildableBackend;

pub mod backend;
pub(crate) mod backend_memory;
pub(crate) mod counting_bitmap;
pub mod result;
pub(crate) mod spaces;
pub(crate) mod unaligned_f32;
pub mod vector;
pub(crate) mod vector_store;

pub use backend_memory::MemoryBackend;
pub use vector_store::VectorStore;

mod helpers;
pub use helpers::*;

use anyhow::Result;

pub use result::ResultSet;

pub type Vector = Vec<f32>;
pub type ID = u64;
pub type Basis = Vec<Vector>;

pub mod bitmaps;
pub use bitmaps::*;

pub fn full_table_scan_search<B: BuildableBackend>(
    backend: &B,
    target: &Vector,
    k: usize,
) -> Result<ResultSet> {
    let mut set = ResultSet::new(k);
    for (id, _) in backend.iter_vecs() {
        let sim = backend.compute_similarity(target, id)?;
        set.add_result(id, sim);
    }
    Ok(set)
}
