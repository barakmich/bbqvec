package bbq

import (
	"fmt"

	"github.com/kelindar/bitmap"
)

type CountingBitmap struct {
	bms []bitmap.Bitmap
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

func (c *CountingBitmap) Add(v uint32) {
	for i := 0; i < len(c.bms); i++ {
		if !c.bms[i].Contains(v) {
			c.bms[i].Set(v)
			break
		}
	}
}

func (c *CountingBitmap) Or(in *bitmap.Bitmap) {
	cur := in.Clone(nil)
	for i := 0; i < len(c.bms); i++ {
		and := cur.Clone(nil)
		and.And(c.bms[i])
		c.bms[i].Or(cur)
		cur = and
	}
}

// TopK may return more things than intended
func (c *CountingBitmap) TopK(k int) []uint32 {
	for i := len(c.bms) - 1; i >= 0; i-- {
		if i != 0 && c.bms[i].Count() < k {
			continue
		}
		arr := make([]uint32, 0, c.bms[i].Count())
		c.bms[i].Range(func(x uint32) {
			arr = append(arr, x)
		})
		return arr
	}
	return nil
}
