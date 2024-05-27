use crate::Bitmap;

#[derive(Default)]
pub struct CountingBitmap<B: Bitmap> {
    bitmaps: Vec<B>,
    a_buf: B,
    b_buf: B,
}

impl<B: Bitmap> CountingBitmap<B> {
    pub fn or(mut self, rhs: &B) {
        rhs.clone_into(&mut self.b_buf);
        let mut cur = rhs;
        let mut next = &mut self.b_buf;
        for i in 0..self.bitmaps.len() {
            next.and(cur);
            next.and(&self.bitmaps[i]);
            self.bitmaps[i].or(cur);
            if i % 2 == 0 {
                cur = &self.b_buf;
                next = &mut self.a_buf;
            } else {
                cur = &self.a_buf;
                next = &mut self.b_buf;
            }
        }
    }

    pub fn top_k(&mut self, search_k: usize) -> Option<&B> {
        self.bitmaps.iter().rev().find(|x| x.count() > search_k)
    }
}
