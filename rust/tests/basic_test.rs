use anyhow::Result;
use bbqvec::{self, IndexIDIterator};

#[test]
fn creates_a_vector() {
    let v = bbqvec::create_random_vector(20);
    assert_eq!(v.len(), 20);
}

#[test]
fn full_table_scan() -> Result<()> {
    let vecs = bbqvec::create_vector_set(20, 200);
    let mem = bbqvec::MemoryBackend::new(20, 3)?;
    let mut store = bbqvec::VectorStore::new_dense_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    let target = bbqvec::create_random_vector(256);
    store.find_nearest(&target, 20, 1000, 16)?;
    Ok(())
}

#[test]
fn built_index() -> Result<()> {
    let vecs = bbqvec::create_vector_set(20, 2000);
    let mem = bbqvec::MemoryBackend::new(20, 10)?;
    let mut store = bbqvec::VectorStore::new_dense_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    store.build_index()?;
    Ok(())
}

#[test]
fn built_big_index() -> Result<()> {
    let vecs = bbqvec::create_vector_set(256, 100000);
    let mem = bbqvec::MemoryBackend::new(256, 30)?;
    let mut store = bbqvec::VectorStore::new_dense_bitmap(mem)?;
    store.add_vector_iter(vecs.enumerate_ids())?;
    store.build_index()?;
    Ok(())
}
