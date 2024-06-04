use bbqvec::Bitmap;
use bitvec::prelude::*;
use std::ops::BitOr;

use criterion::{black_box, Criterion};
use rand::Rng;

pub fn criterion_benchmark_micro(c: &mut Criterion) {
    c.bench_function("create_random_vector_100", |b| {
        b.iter(|| bbqvec::create_random_vector(100))
    });
    c.bench_function("normalize_100", |b| {
        let mut vec = bbqvec::create_random_vector(100);
        b.iter(|| bbqvec::vector::normalize(&mut vec));
    });
    c.bench_function("dot_product_100", |b| {
        let vec = bbqvec::create_random_vector(100);
        let normal = bbqvec::create_random_vector(100);
        b.iter(|| bbqvec::vector::dot_product(&vec, &normal));
    });
    c.bench_function("roaring", |b| {
        let mut x = roaring::RoaringBitmap::new();
        let mut y = roaring::RoaringBitmap::new();
        for _ in 0..20000 {
            x.insert(rand::thread_rng().gen_range(0..2000000));
            y.insert(rand::thread_rng().gen_range(0..2000000));
        }
        b.iter(|| {
            black_box((&x).bitor(&y));
        });
    });
    c.bench_function("bitmap", |b| {
        let mut x = BitVec::<usize, Lsb0>::new();
        let mut y = BitVec::new();
        for _ in 0..20000 {
            x.add(rand::thread_rng().gen_range(0..2000000));
            y.add(rand::thread_rng().gen_range(0..2000000));
        }
        b.iter(|| {
            black_box({
                let mut z = BitVec::new();
                z = z.bitor(&x);
                z = z.bitor(&y);
                z
            });
        });
    });
}
