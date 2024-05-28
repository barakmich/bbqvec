use criterion::{criterion_group, criterion_main};
mod micro;
use micro::criterion_benchmark_micro;
mod memory_store;
use memory_store::criterion_benchmark_memory_store;

criterion_group!(memory, criterion_benchmark_memory_store);
criterion_group!(micro, criterion_benchmark_micro);
criterion_main!(micro, memory);
