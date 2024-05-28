use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark_main(c: &mut Criterion) {
    c.bench_function("create_random_vector 100", |b| {
        b.iter(|| bbqvec::create_random_vector(100))
    });
    c.bench_function("normalize 100", |b| {
        let mut vec = bbqvec::create_random_vector(100);
        b.iter(|| bbqvec::vector::normalize(&mut vec));
    });
    c.bench_function("dot_product 100", |b| {
        let vec = bbqvec::create_random_vector(100);
        let normal = bbqvec::create_random_vector(100);
        b.iter(|| bbqvec::vector::dot_product(&vec, &normal));
    });
    c.bench_function("project_to_plane 100", |b| {
        let mut vec = bbqvec::create_random_vector(100);
        let normal = bbqvec::create_random_vector(100);
        b.iter(|| bbqvec::vector::project_to_plane(&mut vec, &normal));
    });
}

criterion_group!(memory_store, criterion_benchmark_main);
criterion_main!(memory_store);
