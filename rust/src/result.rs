use crate::ID;

#[derive(Debug)]
pub struct SearchResult {
    pub similarity: f32,
    pub id: ID,
}

#[derive(Debug)]
pub struct ResultSet {
    sims: Vec<f32>,
    ids: Vec<ID>,
    k: usize,
    valid: usize,
}

impl ResultSet {
    pub fn new(k: usize) -> Self {
        Self {
            sims: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
            k,
            valid: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.sims.len()
    }

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

    // Consumes the ResultSet and turns it into an ordered list of results
    fn into_iter(self) -> impl Iterator<Item = SearchResult> {
        self.sims
            .into_iter()
            .zip(self.ids)
            .map(|(sim, id)| SearchResult {
                similarity: sim,
                id,
            })
    }
}
