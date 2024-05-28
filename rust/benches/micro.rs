use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark_main(c: &mut Criterion) {
    c.bench_function("create_random_vector 100", |b| {
        b.iter(|| bbqvec::create_random_vector(100))
    });
}

criterion_group!(memory_store, criterion_benchmark_main);
criterion_main!(memory_store);
