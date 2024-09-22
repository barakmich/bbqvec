use std::time::{Duration, Instant};

use anyhow::Result;
use bbqvec::{
    backend::VectorBackend, Bitmap, IndexIDIterator, MemoryBackend, ResultSet, Vector, VectorStore,
};

gflags::define! {
    -v, --vectors: usize = 100000
}

gflags::define! {
    -q, --queries: usize = 1000
}

gflags::define! {
    -d, --dimensions: usize = 256
}

gflags::define! {
    -b, --bases: usize = 30
}

gflags::define! {
    -k, --search-k: usize = 1000
}

gflags::define! {
    -s, --spill: usize = 16
}

enum Mode {
    SingleRun,
    Matrix,
}

fn main() -> Result<()> {
    let args = gflags::parse();
    let mode = if args.is_empty() {
        Mode::SingleRun
    } else {
        match args[0] {
            "run" => Mode::SingleRun,
            "matrix" => Mode::Matrix,
            _ => Mode::SingleRun,
        }
    };
    match mode {
        Mode::SingleRun => single_run_main(),
        Mode::Matrix => matrix_main(),
    }
}

fn single_run_main() -> Result<()> {
    let store = make_store()?;
    let tests = bbqvec::create_vector_set(DIMENSIONS.flag, QUERIES.flag);
    let mut fts_results = Vec::with_capacity(tests.len());
    for t in tests.iter() {
        fts_results.push(store.full_table_scan(t, 20)?);
    }
    let (results, took) = run_test(&tests, &store, SEARCH_K.flag, SPILL.flag)?;
    print_result_line(&fts_results, &results, SEARCH_K.flag, SPILL.flag, took)?;
    Ok(())
}

fn matrix_main() -> Result<()> {
    let store = make_store()?;
    let tests = bbqvec::create_vector_set(DIMENSIONS.flag, QUERIES.flag);
    let mut fts_results = Vec::with_capacity(tests.len());
    for t in tests.iter() {
        fts_results.push(store.full_table_scan(t, 20)?);
    }
    for spill in [1, 4, 8, 16] {
        for searchk in [100, 500, 1000, 2000, 5000, 10000, 20000] {
            if DIMENSIONS.flag < spill {
                continue;
            }
            let (results, took) = run_test(&tests, &store, searchk, spill)?;
            print_result_line(&fts_results, &results, searchk, spill, took)?;
        }
    }
    Ok(())
}

fn make_store() -> Result<VectorStore<MemoryBackend, bbqvec::CRoaringBitmap>> {
    let data = bbqvec::create_vector_set(DIMENSIONS.flag, VECTORS.flag);
    println!("Made vectors");
    let mem = bbqvec::MemoryBackend::new(DIMENSIONS.flag, BASES.flag)?;
    let mut store = bbqvec::VectorStore::new(mem)?;
    store.add_vector_iter(data.enumerate_ids())?;
    println!("Added vectors");
    Ok(store)
}

fn run_test<E: VectorBackend, B: Bitmap>(
    tests: &Vec<Vector>,
    store: &bbqvec::VectorStore<E, B>,
    search_k: usize,
    spill: usize,
) -> Result<(Vec<ResultSet>, Duration)> {
    let mut out = Vec::with_capacity(tests.len());
    let start = Instant::now();
    for v in tests {
        let res = store.find_nearest(v, 20, search_k, spill)?;
        out.push(res);
    }
    let took = Instant::now().duration_since(start);
    Ok((out, took))
}

fn print_result_line(
    fts: &[ResultSet],
    real: &[ResultSet],
    search_k: usize,
    spill: usize,
    took: Duration,
) -> Result<()> {
    let mut acc = [0.0; 4];
    let mut checked = 0;
    for (f, r) in fts.iter().zip(real.iter()) {
        acc[0] += f.compute_recall(r, 1);
        acc[1] += f.compute_recall(r, 5);
        acc[2] += f.compute_recall(r, 10);
        acc[3] += f.compute_recall(r, 20);
        checked += r.checked;
    }
    acc.iter_mut().for_each(|v| *v *= 100.0 / fts.len() as f64);
    let per = took.as_millis() as f64 / real.len() as f64;
    let avg_check = checked / real.len();
    println!(
        "searchk {:<6} / spill {:<4}  ({:8.4}ms, {:10} checked)    {:5.2}@1   {:5.2}@5   {:5.2}@10   {:5.2}@20",
        search_k, spill, per, avg_check, acc[0], acc[1], acc[2], acc[3]
    );
    Ok(())
}
