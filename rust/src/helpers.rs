use rand::Rng;

use crate::{Vector, ID};

#[must_use]
pub fn create_random_vector(dimensions: usize) -> Vector {
    let mut out = Vec::new();
    for _ in 0..dimensions {
        out.push(rand::thread_rng().gen_range(-1.0..1.0));
    }
    crate::vector::normalize(&mut out);
    out
}

#[must_use]
pub fn create_vector_set(dimensions: usize, count: usize) -> Vec<Vector> {
    std::iter::repeat_with(|| create_random_vector(dimensions))
        .take(count)
        .collect()
}

pub trait IndexIDIterator {
    fn enumerate_ids(&self) -> impl Iterator<Item = (ID, &Vector)>;
}

impl IndexIDIterator for Vec<Vector> {
    fn enumerate_ids(&self) -> impl Iterator<Item = (ID, &Vector)> {
        self.iter().enumerate().map(|(i, v)| (i as ID, v))
    }
}
