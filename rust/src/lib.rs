pub(crate) mod counting_bitmap;
pub(crate) mod quantization;
pub(crate) mod vector_file;
pub use quantization::BF16Quantization;
pub use quantization::NoQuantization;

pub mod result;
pub use result::ResultSet;

pub(crate) mod spaces;
pub(crate) mod unaligned_f32;

pub mod backend;
pub use backend::VectorBackend;

pub(crate) mod backend_memory;
pub use backend_memory::MemoryBackend;
pub use backend_memory::QuantizedMemoryBackend;

pub(crate) mod backend_disk;
pub use backend_disk::DiskBackend;

pub mod vector;

pub(crate) mod vector_store;
pub use vector_store::VectorStore;

mod helpers;
pub use helpers::*;

pub type Vector = Vec<f32>;
pub type ID = u64;
pub type Basis = Vec<Vector>;

pub mod bitmaps;
pub use bitmaps::*;
