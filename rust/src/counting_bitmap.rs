use crate::Bitmap;

#[derive(Default)]
pub struct CountingBitmap<B: Bitmap> {
    bitmaps: Vec<B>,
}

impl<B: Bitmap> CountingBitmap<B> {
    pub fn new(size: usize) -> Self {
        Self {
            bitmaps: vec![B::new(); size],
        }
    }

    pub fn or(&mut self, rhs: &B) {
        let mut cur = rhs.clone();
        for i in 0..self.bitmaps.len() {
            self.bitmaps[i].xor(&cur);
            cur.and_not(&self.bitmaps[i]);
            self.bitmaps[i].or(&cur);
            if cur.count() == 0 {
                break;
            }
        }
    }

    pub fn top_k(&self, search_k: usize) -> Option<&B> {
        println!(
            "{:?}",
            self.bitmaps.iter().map(|b| b.count()).collect::<Vec<_>>()
        );
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
        let bm_c = bitvec![usize, Lsb0; 0, 1, 1];
        cbm.or(&bm_a);
        cbm.or(&bm_b);
        cbm.or(&bm_c);
        let v: Vec<u64> = cbm.top_k(1).unwrap().iter_elems().collect();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 2);
        assert_eq!(cbm.top_k(10), None);
    }
}
