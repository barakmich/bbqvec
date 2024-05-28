package bbq

import (
	"fmt"

	"github.com/kelindar/bitmap"
)

type CountingBitmap struct {
	bms []bitmap.Bitmap
	buf bitmap.Bitmap
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
	in.Clone(&c.buf)
	cur := c.buf
	for i := 0; i < len(c.bms); i++ {
		c.bms[i].Xor(cur)
		cur.AndNot(c.bms[i])
		c.bms[i].Or(cur)
		if cur.Count() == 0 {
			break
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
