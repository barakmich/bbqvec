use crate::{vector::distance, Vector};
use anyhow::Result;
use half::{bf16, vec::HalfFloatVecExt};

pub trait Quantization: Default {
    type Lower: Clone;
    fn similarity(x: &Self::Lower, y: &Self::Lower) -> Result<f32>;
    fn compare(x: &Vector, y: &Self::Lower) -> Result<f32>;
    fn lower(vec: Vector) -> Result<Self::Lower>;
    fn vector_size(dimensions: usize) -> usize;
    fn marshal(v: &Self::Lower, array: &mut [u8]) -> Result<()>;
    fn unmarshal(array: &[u8]) -> Result<Self::Lower>;
    fn name() -> &'static str;
}

#[derive(Default)]
pub struct NoQuantization {}

impl Quantization for NoQuantization {
    type Lower = Vector;

    fn similarity(x: &Self::Lower, y: &Self::Lower) -> Result<f32> {
        Ok(distance(x, y))
    }

    fn compare(x: &Vector, y: &Self::Lower) -> Result<f32> {
        Ok(distance(x, y))
    }

    fn lower(vec: Vector) -> Result<Self::Lower> {
        Ok(vec)
    }

    fn name() -> &'static str {
        "none"
    }

    fn vector_size(dimensions: usize) -> usize {
        return 4 * dimensions;
    }

    fn marshal(v: &Self::Lower, array: &mut [u8]) -> Result<()> {
        for (i, f) in v.iter().enumerate() {
            let bytes = f.to_le_bytes();
            &array[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        Ok(())
    }

    fn unmarshal(array: &[u8]) -> Result<Self::Lower> {
        let mut vec = Vec::new();
        for i in (0..array.len()).step_by(4) {
            let bytes = &array[i..i + 4];
            let f: f32 = f32::from_le_bytes(bytes.try_into().unwrap());
            vec.push(f);
        }
        Ok(vec)
    }
}

#[derive(Default)]
pub struct BF16Quantization {}

impl Quantization for BF16Quantization {
    type Lower = Vec<half::bf16>;

    fn similarity(x: &Self::Lower, y: &Self::Lower) -> Result<f32> {
        let fx = x.iter().map(|v| v.to_f32()).collect();
        let fy = y.iter().map(|v| v.to_f32()).collect();
        Ok(distance(&fx, &fy))
    }

    fn compare(x: &Vector, y: &Self::Lower) -> Result<f32> {
        let fy = y.iter().map(|v| v.to_f32()).collect();
        Ok(distance(x, &fy))
    }

    fn lower(vec: Vector) -> Result<Self::Lower> {
        Ok(Vec::from_f32_slice(vec.as_slice()))
    }

    fn name() -> &'static str {
        "bf16"
    }

    fn vector_size(dimensions: usize) -> usize {
        return 2 * dimensions;
    }

    fn marshal(v: &Self::Lower, array: &mut [u8]) -> Result<()> {
        for (i, f) in v.iter().enumerate() {
            let bytes = f.to_le_bytes();
            &array[i * 2..i * 2 + 2].copy_from_slice(&bytes);
        }
        Ok(())
    }

    fn unmarshal(array: &[u8]) -> Result<Self::Lower> {
        let mut vec = Vec::new();
        for i in (0..array.len()).step_by(2) {
            let bytes = &array[i..i + 2];
            let f: bf16 = bf16::from_le_bytes(bytes.try_into().unwrap());
            vec.push(f);
        }
        Ok(vec)
    }
}
