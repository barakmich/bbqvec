use anyhow::{anyhow, Result};
use argminmax::ArgMinMax;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::{
    backend::{BuildableBackend, VectorBackend},
    vector::dot_product,
    Basis, Bitmap, ResultSet, Vector, ID,
};

pub struct VectorStore<E: VectorBackend, B: Bitmap = crate::BitVec> {
    backend: E,
    dimensions: usize,
    n_basis: usize,
    bases: Option<Vec<Basis>>,
    // If we ever have more than INT_MAX_32 dimensions, I quit.
    bitmaps: Option<Vec<HashMap<i32, B>>>,
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

    pub fn find_nearest(
        &self,
        target: Vector,
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
        target: Vector,
        k: usize,
        search_k: usize,
        spill: usize,
    ) -> Result<ResultSet> {
        let rs = ResultSet::new(k);
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

        let bases = Self::make_basis(be, self.n_basis, self.dimensions)?;
        let bitmaps = Self::make_bitmaps(be, &bases)?;
        (&mut self).save_all(&bases, &bitmaps)?;
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
        let prep: Vec<_> = bases.iter().map(|b| (b.clone(), be.iter())).collect();

        prep.into_par_iter()
            .map(|(b, it)| {
                let mut out: HashMap<i32, B> = HashMap::new();
                it.for_each(|(id, vec)| {
                    let mut proj = Vec::new();
                    for basis in b {
                        proj.push(dot_product(vec, &basis));
                    }
                    let (min_idx, max_idx) = proj.argminmax();
                    let idx = if proj[max_idx].abs() >= proj[min_idx].abs() {
                        max_idx as i32
                    } else {
                        min_idx as i32
                    };
                    let face_num = if proj[idx as usize] > 0.0 {
                        idx + 1
                    } else {
                        -(idx + 1)
                    };
                    out.entry(face_num).or_default().add(id);
                });
                Ok(out)
            })
            .collect::<Vec<Result<HashMap<i32, B>>>>()
            .into_iter()
            .collect()
    }

    fn save_all(&mut self, _bases: &Vec<Basis>, _bitmaps: &Vec<HashMap<i32, B>>) -> Result<()> {
        match self.backend.as_indexable_backend() {
            Some(_) => todo!(),
            None => Ok(()),
        }
    }
}

fn create_split(i: usize, basis: &Basis, be: &impl BuildableBackend) -> Result<Vector> {
    todo!()
}
