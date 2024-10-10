//! All the credit to Meilisearch, who did the deep dive here.
//! Their MIT License is also in the `spaces/` directory, and this crate is similarly open-source.
use std::{
    borrow::Borrow,
    fmt,
    mem::{size_of, transmute},
};

use bytemuck::cast_slice;
use byteorder::ByteOrder;

#[allow(clippy::module_name_repetitions)]
/// A wrapper struct that is used to read unaligned floats directly from memory.
#[repr(transparent)]
pub struct UnalignedF32Slice([u8]);

impl UnalignedF32Slice {
    /// Creates an unaligned slice of f32 wrapper from a slice of bytes.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<&Self> {
        if bytes.len() % size_of::<f32>() == 0 {
            #[allow(clippy::transmute_ptr_to_ptr, clippy::missing_transmute_annotations)]
            Ok(unsafe { transmute(bytes) })
        } else {
            Err(anyhow::anyhow!("Byte size mismatch to f32"))
        }
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    pub fn from_slice(slice: &[f32]) -> &Self {
        Self::from_bytes(cast_slice(slice)).unwrap()
    }

    /// Returns the original raw slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Return the number of f32 that fits into this slice.
    pub fn len(&self) -> usize {
        self.0.len() / size_of::<f32>()
    }

    /// Returns wether it is empty or not.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator of f32 that are read from the slice.
    /// The f32 are copied in memory and are therefore, aligned.
    #[allow(clippy::needless_lifetimes)]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.0
            .chunks_exact(size_of::<f32>())
            .map(byteorder::NativeEndian::read_f32)
    }

    /// Returns the raw pointer to the start of this slice.
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

impl<'a> From<&'a crate::Vector> for &'a UnalignedF32Slice {
    fn from(value: &'a Vec<f32>) -> Self {
        UnalignedF32Slice::from_slice(value)
    }
}

impl ToOwned for UnalignedF32Slice {
    type Owned = Vec<f32>;

    fn to_owned(&self) -> Self::Owned {
        bytemuck::pod_collect_to_vec(&self.0)
    }
}

impl Borrow<UnalignedF32Slice> for Vec<f32> {
    fn borrow(&self) -> &UnalignedF32Slice {
        UnalignedF32Slice::from_slice(&self[..])
    }
}

impl fmt::Debug for UnalignedF32Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct SmallF32(f32);
        impl fmt::Debug for SmallF32 {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_fmt(format_args!("{:.4?}", self.0))
            }
        }

        let mut list = f.debug_list();
        self.iter().for_each(|float| {
            list.entry(&SmallF32(float));
        });
        list.finish()
    }
}
