use crate::Vector;

#[allow(clippy::inline_always)]
#[inline(always)]
pub fn normalize(vec: &mut Vector) {
    let s = crate::unaligned_f32::UnalignedF32Slice::from_slice(vec.as_slice());
    let norm = crate::spaces::simple::dot_product(s, s).sqrt();
    vec.iter_mut().for_each(|v| *v /= norm);
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[must_use]
pub fn dot_product(vec: &Vector, other: &Vector) -> f32 {
    crate::spaces::simple::dot_product(vec.into(), other.into())
}

#[allow(clippy::inline_always)]
#[inline(always)]
pub fn subtract_into(vec: &mut Vector, other: &Vector) {
    for (v, o) in vec.iter_mut().zip(other.iter()) {
        *v -= o;
    }
}

#[allow(clippy::inline_always)]
#[inline(always)]
#[must_use]
pub fn distance(vec: &Vector, other: &Vector) -> f32 {
    vec.iter()
        .zip(other.iter())
        .fold(0.0, |acc, (a, b)| acc + ((a - b) * (a - b)))
        .sqrt()
}
