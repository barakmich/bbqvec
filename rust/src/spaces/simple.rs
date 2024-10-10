#[allow(clippy::wildcard_imports)]
#[cfg(target_arch = "x86_64")]
use super::simple_avx::*;
#[allow(clippy::wildcard_imports)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::simple_neon::*;
#[allow(clippy::wildcard_imports)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::simple_sse::*;
use crate::unaligned_f32::UnalignedF32Slice;

#[cfg(target_arch = "x86_64")]
const MIN_DIM_SIZE_AVX: usize = 32;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
const MIN_DIM_SIZE_SIMD: usize = 16;

#[allow(dead_code)]
pub fn euclidean_distance(u: &UnalignedF32Slice, v: &UnalignedF32Slice) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
            && u.len() >= MIN_DIM_SIZE_AVX
        {
            return unsafe { euclid_similarity_avx(u, v) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse") && u.len() >= MIN_DIM_SIZE_SIMD {
            return unsafe { euclid_similarity_sse(u, v) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && u.len() >= MIN_DIM_SIZE_SIMD {
            return unsafe { euclid_similarity_neon(u, v) };
        }
    }

    euclidean_distance_non_optimized(u, v)
}

// Don't use dot-product: avoid catastrophic cancellation in
// https://github.com/spotify/annoy/issues/314.
pub fn euclidean_distance_non_optimized(u: &UnalignedF32Slice, v: &UnalignedF32Slice) -> f32 {
    u.iter().zip(v.iter()).map(|(u, v)| (u - v) * (u - v)).sum()
}

pub fn dot_product(u: &UnalignedF32Slice, v: &UnalignedF32Slice) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
            && u.len() >= MIN_DIM_SIZE_AVX
        {
            return unsafe { dot_similarity_avx(u, v) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse") && u.len() >= MIN_DIM_SIZE_SIMD {
            return unsafe { dot_similarity_sse(u, v) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && u.len() >= MIN_DIM_SIZE_SIMD {
            return unsafe { dot_similarity_neon(u, v) };
        }
    }

    dot_product_non_optimized(u, v)
}

pub fn dot_product_non_optimized(u: &UnalignedF32Slice, v: &UnalignedF32Slice) -> f32 {
    u.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}
