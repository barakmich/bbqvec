package bbq

import (
	"fmt"

	"github.com/kelindar/bitmap"
)

type CountingBitmap struct {
	bms  []bitmap.Bitmap
	and  bitmap.Bitmap
	andb bitmap.Bitmap
}

func NewCountingBitmap(maxCount int) *CountingBitmap {
	return &CountingBitmap{
		bms: make([]bitmap.Bitmap, maxCount),
	}
}

func (c *CountingBitmap) cardinalities() []int {
	cards := make([]int, len(c.bms))
	for i, it := range c.bms {
		cards[i] = it.Count()
	}
	return cards
}

func (c *CountingBitmap) String() string {
	return fmt.Sprint(c.cardinalities())
}

func (c *CountingBitmap) Or(in bitmap.Bitmap) {
	c.and = in
	in.Clone(&c.andb)
	for i := 0; i < len(c.bms); i++ {
		if i%2 == 0 {
			if c.and.Count() == 0 {
				break
			}
			c.andb.And(c.and, c.bms[i])
			c.bms[i].Or(c.and)
		} else {
			if c.andb.Count() == 0 {
				break
			}
			c.and.And(c.andb, c.bms[i])
			c.bms[i].Or(c.andb)
		}
	}
}

// TopK may return more things than intended
func (c *CountingBitmap) TopK(k int) bitmap.Bitmap {
	for i := len(c.bms) - 1; i >= 0; i-- {
		if i != 0 && c.bms[i].Count() < k {
			continue
		}
		return c.bms[i]
	}
	return nil
}
