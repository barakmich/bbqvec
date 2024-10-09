use crate::ID;

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct SearchResult {
    pub similarity: f32,
    pub id: ID,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct ResultSet {
    sims: Vec<f32>,
    ids: Vec<ID>,
    k: usize,
    pub checked: usize,
}

impl ResultSet {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            sims: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
            k,
            checked: 0,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sims.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sims.is_empty()
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
    #[must_use]
    pub fn compute_recall(&self, baseline: &ResultSet, at: usize) -> f64 {
        let mut found = 0;
        for x in baseline.ids.iter().take(at) {
            for y in self.ids.iter().take(at) {
                if x == y {
                    found += 1;
                }
            }
        }
        found as f64 / at as f64
    }

    pub fn add_result(&mut self, id: ID, similarity: f32) {
        self.checked += 1;
        if self.sims.len() == self.k {
            let last = self.sims.last().unwrap_or(&f32::MIN);
            if *last > similarity {
                return;
            }
        }
        let mut insert: usize = 0;
        let mut found: bool = false;
        for (self_id, self_sim) in self.ids.iter().zip(self.sims.iter()) {
            if id == *self_id {
                // Found ourselves
                return;
            }
            if *self_sim < similarity {
                found = true;
                break;
            }
            insert += 1;
        }
        if !found {
            // Last chance -- we're not big enough yet, so you can join the club.
            if self.sims.len() < self.k {
                self.sims.push(similarity);
                self.ids.push(id);
            }
            return;
        }
        self.ids.insert(insert, id);
        self.ids.truncate(self.k);
        self.sims.insert(insert, similarity);
        self.sims.truncate(self.k);
    }

    pub fn iter_results(&self) -> impl Iterator<Item = SearchResult> + '_ {
        self.sims
            .iter()
            .zip(self.ids.iter())
            .map(|(sim, id)| SearchResult {
                similarity: *sim,
                id: *id,
            })
    }
}
