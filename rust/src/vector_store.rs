use anyhow::{anyhow, Result};
use argminmax::ArgMinMax;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::{
    backend::{BuildableBackend, VectorBackend},
    counting_bitmap::CountingBitmap,
    vector::dot_product,
    Basis, Bitmap, IndexIDIterator, ResultSet, Vector, ID,
};

pub struct VectorStore<E: VectorBackend, B: Bitmap> {
    backend: E,
    dimensions: usize,
    n_basis: usize,
    bases: Option<Vec<Basis>>,
    // If we ever have more than INT_MAX_32 dimensions, I quit.
    bitmaps: Option<Vec<HashMap<i32, B>>>,
}

impl<E: VectorBackend> VectorStore<E, crate::BitVec> {
    pub fn new_dense_bitmap(backend: E) -> Result<Self> {
        VectorStore::new(backend)
    }
}

impl<E: VectorBackend, B: Bitmap> VectorStore<E, B> {
    pub fn new(backend: E) -> Result<Self> {
        let info = backend.info();
        let mut out = Self {
            backend,
            dimensions: info.dimensions,
            n_basis: info.n_basis,
            bases: None,
            bitmaps: None,
        };
        if info.has_index_data {
            out.load_from_backend()?;
        }
        Ok(out)
    }

    fn load_from_backend(&mut self) -> Result<()> {
        Ok(())
    }

    #[inline(always)]
    pub fn add_vector(&mut self, id: ID, vector: &Vector) -> Result<()> {
        if self.bitmaps.is_some() {
            // TODO: Add vectors after building to the bitmaps
            Err(anyhow!("already built"))
        } else {
            self.backend.put_vector(id, vector)
        }
    }

    pub fn add_vector_iter<'a>(
        &mut self,
        iter: impl Iterator<Item = (ID, &'a Vector)>,
    ) -> Result<()> {
        for (id, vec) in iter {
            self.add_vector(id, vec)?;
        }
        Ok(())
    }

    pub fn find_nearest(
        &self,
        target: &Vector,
        k: usize,
        search_k: usize,
        spill: usize,
    ) -> Result<ResultSet> {
        if self.bitmaps.is_none() {
            match self.backend.as_buildable_backend() {
                Some(be) => crate::full_table_scan_search(be, target, k),
                None => panic!("Backend cannot be built?"),
            }
        } else {
            let sp = if spill >= self.dimensions {
                self.dimensions - 1
            } else {
                spill
            };
            self.find_nearest_internal(target, k, search_k, sp)
        }
    }

    #[inline(always)]
    fn find_nearest_internal(
        &self,
        target: &Vector,
        k: usize,
        search_k: usize,
        spill: usize,
    ) -> Result<ResultSet> {
        let mut rs = ResultSet::new(k);
        let mut bs = CountingBitmap::<B>::new(self.bases.as_ref().unwrap().len());
        let mut proj: Vec<f32> = Vec::with_capacity(self.dimensions);
        for (i, basis) in self.bases.as_ref().unwrap().iter().enumerate() {
            let mut spill_into = B::new();
            proj.clear();
            for b in basis {
                proj.push(dot_product(target, b))
            }
            for _s in 0..(spill + 1) {
                let face_idx = find_face_idx(&proj);
                if let Some(bm) = self.bitmaps.as_ref().unwrap()[i].get(&face_idx) {
                    spill_into.or(bm);
                };
                proj[face_idx.unsigned_abs() as usize] = 0.0
            }
            bs.or(&spill_into);
        }
        let elems = bs
            .top_k(search_k)
            .ok_or(anyhow!("Didn't find a counting layer?"))?;
        for id in elems.iter_elems() {
            let sim = self.backend.compute_similarity(target, id)?;
            rs.add_result(id, sim);
        }
        Ok(rs)
    }

    // Consumes the index, returning a new one (likely itself)
    pub fn build_index(mut self) -> Result<VectorStore<E::CompiledBackend, B>> {
        if self.bitmaps.is_some() {
            return Err(anyhow!("Already built"));
        }
        let be = self
            .backend
            .as_buildable_backend()
            .ok_or(anyhow!("Backend not buildable"))?;

        let mut start = Instant::now();
        let bases = Self::make_basis(be, self.n_basis, self.dimensions)?;
        println!("made basis {:?}", Instant::now().duration_since(start));
        start = Instant::now();
        let bitmaps = Self::make_bitmaps(be, &bases)?;
        println!("made bitmaps {:?}", Instant::now().duration_since(start));
        self.save_all(&bases, &bitmaps)?;
        let backend = self.backend.compile()?;
        Ok(VectorStore {
            backend,
            dimensions: self.dimensions,
            n_basis: self.n_basis,
            bases: Some(bases),
            bitmaps: Some(bitmaps),
        })
    }

    fn make_basis(be: &E::Buildable, n_basis: usize, dimensions: usize) -> Result<Vec<Basis>> {
        let mut bases = Vec::<Basis>::with_capacity(n_basis);
        for _n in 0..n_basis {
            let mut basis = Basis::with_capacity(dimensions);
            for i in 0..dimensions {
                let norm = create_split(i, &basis, be)?;
                basis.push(norm);
            }
            bases.push(basis);
        }
        Ok(bases)
    }

    fn make_bitmaps(be: &E::Buildable, bases: &[Basis]) -> Result<Vec<HashMap<i32, B>>> {
        let arc_be = Arc::new(Mutex::new(be));
        let prep: Vec<_> = bases.iter().map(|b| (b.clone(), arc_be.clone())).collect();

        prep.par_iter()
            .map(|(b, arc)| {
                let mut out: HashMap<i32, B> = HashMap::new();
                let it = { arc.lock().unwrap().iter_vecs() };
                it.for_each(|(id, vec)| {
                    let mut proj = Vec::new();
                    for basis in b {
                        proj.push(dot_product(vec, basis));
                    }
                    let face_idx = find_face_idx(&proj);
                    out.entry(face_idx).or_default().add(id);
                });
                Ok(out)
            })
            .collect::<Vec<Result<HashMap<i32, B>>>>()
            .into_iter()
            .collect()
    }

    fn save_all(&mut self, _bases: &[Basis], _bitmaps: &[HashMap<i32, B>]) -> Result<()> {
        match self.backend.as_indexable_backend() {
            Some(_) => todo!(),
            None => Ok(()),
        }
    }
}

const VECTORS_TO_CONSIDER: usize = 200;

fn create_split(i: usize, basis: &Basis, be: &impl BuildableBackend) -> Result<Vector> {
    // First, find a random vector in the set
    let mut p = pick_random_vec(i, basis, be)?;
    let (mut q, _dist) = (0..VECTORS_TO_CONSIDER).try_fold(
        (pick_random_vec(i, basis, be)?, -2.0),
        |(vector, distance), _| {
            let candidate = pick_random_vec(i, basis, be)?;
            let dist = crate::vector::distance(&candidate, &vector);
            if dist > distance {
                anyhow::Ok((candidate, dist))
            } else {
                anyhow::Ok((vector, distance))
            }
        },
    )?;
    crate::vector::normalize(&mut p);
    crate::vector::normalize(&mut q);
    crate::vector::subtract_into(&mut p, &q);
    crate::vector::normalize(&mut p);
    Ok(p)
}

fn pick_random_vec(depth: usize, basis: &Basis, be: &impl BuildableBackend) -> Result<Vector> {
    let mut v = be.get_random_vector()?.clone();
    reduce_vector(&mut v, depth, basis);
    Ok(v)
}

#[inline(always)]
fn reduce_vector(vector: &mut Vector, depth: usize, basis: &Basis) {
    basis
        .iter()
        .take(depth)
        .for_each(|b| crate::vector::project_to_plane(vector, b));
}

#[inline(always)]
fn find_face_idx(projection: &Vector) -> i32 {
    let (min_idx, max_idx) = projection.argminmax();
    let idx = if projection[max_idx].abs() >= projection[min_idx].abs() {
        max_idx as i32
    } else {
        min_idx as i32
    };
    if projection[idx as usize] > 0.0 {
        idx + 1
    } else {
        -(idx + 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::MemoryBackend;

    fn vecs() -> Vec<Vector> {
        vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![-1.0, 0.0],
        ]
    }

    #[test]
    fn test_create_basis() {
        let mut mem = MemoryBackend::new(2, 1).unwrap();
        mem.set_rng(Box::new(rand::rngs::mock::StepRng::new(0, 3)));
        for (i, v) in vecs().enumerate_ids() {
            mem.put_vector(i, v).unwrap();
        }
        let be = mem.as_buildable_backend().unwrap();
        let basis_set: Vec<Basis> =
            VectorStore::<MemoryBackend, crate::BitVec>::make_basis(be, 1, 2).unwrap();
        assert_eq!(basis_set.len(), 1);
        assert_eq!(basis_set[0].len(), 2);
        assert_eq!(basis_set[0][0], vec![0.0, 0.0]);
        assert_eq!(basis_set[0][1], vec![0.0, 0.0]);
    }

    #[test]
    fn test_make_bitmaps() {
        //let mem = MemoryBackend::new(2, 2);
    }
}
