use anyhow::Result;
use bbqvec::{self, backend::VectorBackend, IndexIDIterator};

#[test]
fn creates_a_vector() {
    let v = bbqvec::create_random_vector(20);
    assert_eq!(v.len(), 20);
}

#[test]
fn full_table_scan() -> Result<()> {
    let vecs = bbqvec::create_vector_set(20, 200);
    let mut mem = bbqvec::MemoryBackend::new(20, 3)?;
    for (id, v) in vecs.enumerate_ids() {
        mem.put_vector(id, v)?;
    }
    let target = bbqvec::create_random_vector(256);
    let _ = mem.find_nearest(&target, 20)?;
    Ok(())
}

#[test]
fn built_index() -> Result<()> {
    let vecs = bbqvec::create_vector_set(20, 2000);
    let mem = bbqvec::MemoryBackend::new(20, 10)?;
    let mut store = bbqvec::VectorStore::new_croaring_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    Ok(())
}

#[test]
fn built_quantized_index() -> Result<()> {
    let vecs = bbqvec::create_vector_set(20, 2000);
    let mem = bbqvec::QuantizedMemoryBackend::<bbqvec::BF16Quantization>::new(20, 10)?;
    let mut store = bbqvec::VectorStore::new_croaring_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    Ok(())
}

#[test]
fn built_big_index() -> Result<()> {
    let vecs = bbqvec::create_vector_set(256, 1000);
    let mem = bbqvec::MemoryBackend::new(256, 10)?;
    let mut store = bbqvec::VectorStore::new_croaring_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    Ok(())
}
