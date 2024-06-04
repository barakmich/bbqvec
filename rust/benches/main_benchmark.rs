use criterion::{criterion_group, criterion_main, Criterion};
mod micro;
use micro::criterion_benchmark_micro;
mod memory_store;
use memory_store::criterion_benchmark_memory_store;
use pprof::criterion::{Output, PProfProfiler};

criterion_group! {
    name = memory;
    config = Criterion::default().with_profiler(PProfProfiler::new(1000, Output::Protobuf));
    targets = criterion_benchmark_memory_store
}
criterion_group!(micro, criterion_benchmark_micro);
criterion_main!(micro, memory);
