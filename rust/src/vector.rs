use crate::Vector;

#[inline(always)]
pub(crate) fn project_to_plane(vec: &mut Vector, normal: &Vector) {
    let dot = dot_product(vec.as_ref(), normal);
    for (n, v) in normal.iter().zip(vec.iter_mut()) {
        *v -= n * dot
    }
    normalize(vec);
}

#[inline(always)]
pub(crate) fn normalize(vec: &mut Vector) {
    let s = crate::unaligned_f32::UnalignedF32Slice::from_slice(vec.as_slice());
    let norm = crate::spaces::simple::dot_product(s, s).sqrt();
    vec.iter_mut().for_each(|v| *v /= norm);
}

#[inline(always)]
pub(crate) fn dot_product(vec: &Vector, other: &Vector) -> f32 {
    crate::spaces::simple::dot_product(vec.into(), other.into())
}
