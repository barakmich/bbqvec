use anyhow::Result;
use bbqvec::{self, IndexIDIterator};

#[test]
fn search_index() -> Result<()> {
    let data = bbqvec::create_vector_set(10, 100_000);
    let mem = bbqvec::MemoryBackend::new(10, 10)?;
    let mut store = bbqvec::VectorStore::new_croaring_bitmap(mem)?;
    println!("Made store");
    store.add_vector_iter(data.enumerate_ids())?;
    println!("itered");
    println!("built");
    for _i in 0..1000 {
        let target = bbqvec::create_random_vector(10);
        store.find_nearest(&target, 20, 1000, 1)?;
    }
    Ok(())
}
