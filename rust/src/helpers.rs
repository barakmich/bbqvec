use rand::Rng;

use crate::Vector;

pub fn create_random_vector(dimensions: usize) -> Vector {
    let mut out = Vec::new();
    for _ in 0..dimensions {
        out.push(rand::thread_rng().gen_range(0.0..1.0))
    }
    crate::vector::normalize(&mut out);
    out
}

pub fn create_vector_set(dimensions: usize, count: usize) -> Vec<Vector> {
    std::iter::repeat_with(|| create_random_vector(dimensions))
        .take(count)
        .collect()
}
