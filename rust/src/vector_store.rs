use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::{
    backend::{BuildableBackend, VectorBackend},
    Basis, Bitmap, ResultSet, Vector, ID,
};

pub struct VectorStore<E: VectorBackend, B: Bitmap = crate::BitVec> {
    backend: E,
    dimensions: usize,
    n_basis: usize,
    bases: Option<Vec<Basis>>,
    bitmaps: Option<HashMap<usize, B>>,
    built: bool,
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
            built: false,
        };
        if info.has_index_data {
            out.load_from_backend()?;
        }
        Ok(out)
    }

    fn load_from_backend(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn add_vector(&mut self, id: ID, vector: &Vector) -> Result<()> {
        if self.built {
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
        if !self.built {
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
        if self.built {
            return Err(anyhow!("Already built"));
        }
        let be = self
            .backend
            .as_buildable_backend_mut()
            .ok_or(anyhow!("Backend not buildable"))?;
        let bases = Self::make_basis(be, self.n_basis, self.dimensions)?;
        let bitmaps = Self::make_bitmaps(be, &bases, self.n_basis, self.dimensions)?;
        self.save_all(&bases, &bitmaps)?;
        let backend = self.backend.compile()?;
        Ok(VectorStore {
            backend,
            dimensions: self.dimensions,
            n_basis: self.n_basis,
            bases: Some(bases),
            bitmaps: Some(bitmaps),
            built: true,
        })
    }

    fn make_basis(
        be: &mut impl BuildableBackend,
        n_basis: usize,
        dimensions: usize,
    ) -> Result<Vec<Basis>> {
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

    fn make_bitmaps(
        be: &mut impl BuildableBackend,
        bases: &[Basis],
        n_basis: usize,
        dimensions: usize,
    ) -> Result<HashMap<usize, B>> {
        todo!()
    }

    fn save_all(&self, bases: &Vec<Basis>, bitmaps: &HashMap<usize, B>) -> Result<()> {
        todo!()
    }
}

fn create_split(i: usize, basis: &Basis, be: &mut impl BuildableBackend) -> Result<Vector> {
    todo!()
}
