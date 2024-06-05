package bbq

import (
	"fmt"

	"github.com/RoaringBitmap/roaring"
)

type CountingBitmap struct {
	bms []*roaring.Bitmap
}

func NewCountingBitmap(maxCount int) *CountingBitmap {
	return &CountingBitmap{
		bms: make([]*roaring.Bitmap, maxCount),
	}
}

func (c *CountingBitmap) cardinalities() []int {
	cards := make([]int, len(c.bms))
	for i, it := range c.bms {
		cards[i] = int(it.GetCardinality())
	}
	return cards
}

func (c *CountingBitmap) String() string {
	return fmt.Sprint(c.cardinalities())
}

func (c *CountingBitmap) Or(in *roaring.Bitmap) {
	cur := in
	for i := 0; i < len(c.bms); i++ {
		if c.bms[i] == nil {
			c.bms[i] = roaring.NewBitmap()
		}
		c.bms[i].Xor(cur)
		cur.AndNot(c.bms[i])
		c.bms[i].Or(cur)
		if cur.GetCardinality() == 0 {
			break
		}
	}
}

// TopK may return more things than intended
func (c *CountingBitmap) TopK(k int) *roaring.Bitmap {
	for i := len(c.bms) - 1; i >= 0; i-- {
		if c.bms[i] == nil {
			continue
		}
		if i != 0 && int(c.bms[i].GetCardinality()) < k {
			continue
		}
		return c.bms[i]
	}
	return nil
}
