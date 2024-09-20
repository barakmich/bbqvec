use crate::{vector::distance, Vector};
use anyhow::Result;
use half::vec::HalfFloatVecExt;

pub trait Quantization {
    type Lower: Clone;
    fn similarity(x: &Self::Lower, y: &Self::Lower) -> Result<f32>;
    fn compare(x: &Vector, y: &Self::Lower) -> Result<f32>;
    fn lower(vec: Vector) -> Result<Self::Lower>;
    fn name() -> &'static str;
}

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
}

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
}
