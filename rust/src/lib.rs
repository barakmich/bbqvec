pub mod backend;
pub(crate) mod backend_memory;
pub(crate) mod counting_bitmap;
pub(crate) mod quantization;
pub mod result;
pub(crate) mod spaces;
pub(crate) mod unaligned_f32;
pub mod vector;
pub(crate) mod vector_store;

pub use backend_memory::MemoryBackend;
pub use backend_memory::QuantizedMemoryBackend;
pub use vector_store::VectorStore;

mod helpers;
pub use helpers::*;

pub use result::ResultSet;

pub type Vector = Vec<f32>;
pub type ID = u64;
pub type Basis = Vec<Vector>;

pub mod bitmaps;
pub use bitmaps::*;
