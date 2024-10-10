use std::fmt::Display;

use crate::Bitmap;

#[derive(Default)]
pub struct CountingBitmap<B: Bitmap> {
    bitmaps: Vec<B>,
}

impl<B: Bitmap> Display for CountingBitmap<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.cardinalities()))
    }
}

impl<B: Bitmap> CountingBitmap<B> {
    pub fn new(size: usize) -> Self {
        Self {
            bitmaps: vec![B::new(); size],
        }
    }

    pub fn or(&mut self, rhs: B) {
        let mut cur = rhs;
        for i in 0..self.bitmaps.len() {
            self.bitmaps[i].xor(&cur);
            cur.and_not(&self.bitmaps[i]);
            self.bitmaps[i].or(&cur);
            if cur.is_empty() {
                break;
            }
        }
    }

    pub fn cardinalities(&self) -> Vec<usize> {
        self.bitmaps
            .iter()
            .map(super::bitmaps::Bitmap::count)
            .collect::<Vec<_>>()
    }

    pub fn top_k(&self, search_k: usize) -> Option<&B> {
        self.bitmaps.iter().rev().find(|x| x.count() >= search_k)
    }
}

#[cfg(test)]
mod test {
    use bitvec::prelude::*;

    use super::*;

    #[test]
    fn finds_count() {
        let mut cbm = CountingBitmap::<crate::BitVec>::new(3);
        let bm_a = bitvec![usize, Lsb0; 0, 0, 1];
        let bm_b = bitvec![usize, Lsb0; 0, 1, 1];
        let bm_c = bitvec![usize, Lsb0; 1, 1, 1];
        cbm.or(bm_a);
        cbm.or(bm_b);
        cbm.or(bm_c);
        let v: Vec<u64> = cbm.top_k(1).unwrap().iter_elems().collect();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 2);
        assert_eq!(cbm.top_k(10), None);
        assert_eq!(cbm.top_k(1).unwrap().count(), 1);
    }
}
