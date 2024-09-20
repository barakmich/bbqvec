use bbqvec::IndexIDIterator;
use criterion::{BenchmarkId, Criterion};

pub fn criterion_benchmark_memory_store(c: &mut Criterion) {
    let data = bbqvec::create_vector_set(256, 1000000);
    println!("Made vecs");
    let mem = bbqvec::MemoryBackend::new(256, 30).unwrap();
    let mut store = bbqvec::VectorStore::new_croaring_bitmap(mem).unwrap();
    println!("Made store");
    store.add_vector_iter(data.enumerate_ids()).unwrap();
    println!("itered");
    println!("built");
    c.bench_with_input(BenchmarkId::new("find_nearest", "store"), &store, |b, s| {
        b.iter(|| {
            let target = bbqvec::create_random_vector(256);
            s.find_nearest(&target, 20, 1000, 4).unwrap();
        })
    });
}
