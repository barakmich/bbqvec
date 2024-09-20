use anyhow::{anyhow, Result};
use argminmax::ArgMinMax;
use std::{borrow::BorrowMut, collections::HashMap};

use crate::{
    backend::VectorBackend,
    counting_bitmap::CountingBitmap,
    create_random_vector,
    vector::{dot_product, normalize},
    Basis, Bitmap, ResultSet, Vector, ID,
};

pub struct VectorStore<E: VectorBackend, B: Bitmap> {
    backend: E,
    dimensions: usize,
    bases: Vec<Basis>,
    // If we ever have more than INT_MAX_32 dimensions, I quit.
    bitmaps: Vec<HashMap<i32, B>>,
}

impl<E: VectorBackend> VectorStore<E, crate::bitmaps::CRoaringBitmap> {
    pub fn new_croaring_bitmap(backend: E) -> Result<Self> {
        VectorStore::new(backend)
    }
}

impl<E: VectorBackend> VectorStore<E, crate::bitmaps::RoaringBitmap> {
    pub fn new_roaring_bitmap(backend: E) -> Result<Self> {
        VectorStore::new(backend)
    }
}

impl<E: VectorBackend> VectorStore<E, crate::bitmaps::BitVec> {
    pub fn new_bitvec_bitmap(backend: E) -> Result<Self> {
        VectorStore::new(backend)
    }
}

impl<E: VectorBackend, B: Bitmap> VectorStore<E, B> {
    pub fn new(mut backend: E) -> Result<Self> {
        let info = backend.info();
        let bases = match backend.load_bases()? {
            Some(b) => b,
            None => make_basis(info.n_basis, info.dimensions)?,
        };
        let bitmaps = load_all_bitmaps(backend.borrow_mut())?;
        let out = Self {
            backend,
            dimensions: info.dimensions,
            bases,
            bitmaps,
        };
        Ok(out)
    }

    #[inline(always)]
    pub fn add_vector(&mut self, id: ID, vector: &Vector) -> Result<()> {
        self.add_vector_iter(vec![(id, vector)].into_iter())
    }

    pub fn add_vector_iter<'a>(
        &mut self,
        iter: impl Iterator<Item = (ID, &'a Vector)>,
    ) -> Result<()> {
        for (id, vec) in iter {
            self.backend.put_vector(id, vec)?;
            self.add_to_bitmaps(id, vec)?;
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
        let sp = if spill >= self.dimensions {
            self.dimensions - 1
        } else {
            spill
        };
        self.find_nearest_internal(target, k, search_k, sp)
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
        let mut bs = CountingBitmap::<B>::new(self.bases.len());
        let mut proj: Vec<f32> = Vec::with_capacity(self.dimensions);
        for (i, basis) in self.bases.iter().enumerate() {
            let mut spill_into = B::new();
            proj.clear();
            for b in basis {
                proj.push(dot_product(target, b))
            }
            for _s in 0..(spill + 1) {
                let face_idx = find_face_idx(&proj);
                if let Some(bm) = self.bitmaps[i].get(&face_idx) {
                    spill_into.or(bm);
                };
                proj[(face_idx.unsigned_abs() - 1) as usize] = 0.0;
            }
            bs.or(spill_into);
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

    #[allow(unused)]
    fn add_to_bitmaps(&mut self, id: ID, vec: &Vector) -> Result<()> {
        let mut proj = Vec::with_capacity(self.dimensions);
        for (bi, basis) in self.bases.iter().enumerate() {
            proj.clear();
            for b in basis {
                proj.push(dot_product(vec, b));
            }
            let face_idx = find_face_idx(&proj);
            self.bitmaps[bi].entry(face_idx).or_default().add(id);
        }
        Ok(())
    }

    pub fn full_table_scan(&self, vec: &Vector, k: usize) -> Result<ResultSet> {
        self.backend.find_nearest(vec, k)
    }
}

fn make_basis(n_basis: usize, dimensions: usize) -> Result<Vec<Basis>> {
    let mut bases = Vec::<Basis>::with_capacity(n_basis);
    for _n in 0..n_basis {
        let mut basis = Basis::with_capacity(dimensions);
        for _ in 0..dimensions {
            basis.push(create_random_vector(dimensions));
        }
        let out = orthonormalize(basis, 1);
        bases.push(out);
    }
    Ok(bases)
}

#[allow(unused)]
fn print_basis(basis: &Basis) {
    for i in 0..basis.len() {
        for j in 0..basis.len() {
            print!("{:+.4} ", dot_product(&basis[i], &basis[j]));
        }
        println!();
    }
}

fn orthonormalize(mut basis: Basis, rounds: usize) -> Basis {
    let dim = basis[0].len();
    for _ in 0..rounds {
        for i in 0..basis.len() {
            normalize(&mut basis[i]);
            for j in i + 1..basis.len() {
                let dot = dot_product(&basis[i], &basis[j]);
                for k in 0..dim {
                    basis[j][k] -= dot * basis[i][k];
                }
                normalize(&mut basis[j]);
            }
        }
    }
    basis
}

fn load_all_bitmaps<B: Bitmap>(be: &mut impl VectorBackend) -> Result<Vec<HashMap<i32, B>>> {
    let info = be.info();
    let mut out = Vec::with_capacity(info.n_basis);
    for i in 0..info.n_basis {
        let mut hm = HashMap::<i32, B>::new();
        for x in 0..info.dimensions {
            let index = x as i32;
            let bit = be.load_bitmap::<B>(i, index)?;
            if let Some(bitmap) = bit {
                hm.insert(index, bitmap);
            } else {
                hm.insert(index, B::default());
            }
            let bit = be.load_bitmap::<B>(i, -index)?;
            if let Some(bitmap) = bit {
                hm.insert(-index, bitmap);
            } else {
                hm.insert(-index, B::default());
            }
        }
        out.push(hm)
    }
    Ok(out)
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
    use crate::{IndexIDIterator, MemoryBackend};

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
        let basis_set: Vec<Basis> = make_basis(1, 2).unwrap();
        assert_eq!(basis_set.len(), 1);
        assert_eq!(basis_set[0].len(), 2);
    }

    #[test]
    fn test_make_bitmaps() {
        //let mem = MemoryBackend::new(2, 2);
    }
}
