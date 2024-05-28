package bbq

import (
	"math/rand"
	"testing"

	"github.com/RoaringBitmap/roaring"
	"github.com/kelindar/bitmap"
	"github.com/viterin/vek/vek32"
)

func BenchmarkMicroProjection(b *testing.B) {
	buf := make([]float32, 100)
	for i := 0; i < b.N; i++ {
		v := NewRandVector(100, nil)
		n := NewRandVector(100, nil)
		v.projectToPlane(n, buf)
	}
}

func BenchmarkMicroDot(b *testing.B) {
	v := NewRandVector(100, nil)
	n := NewRandVector(100, nil)
	for i := 0; i < b.N; i++ {
		vek32.Dot(v, n)
	}
}

func BenchmarkMicroRoaring(b *testing.B) {
	x := roaring.NewBitmap()
	y := roaring.NewBitmap()
	for range 20000 {
		x.AddInt(rand.Intn(2000000))
		y.AddInt(rand.Intn(2000000))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		roaring.Or(x, y)
	}
}

func BenchmarkMicroBitmap(b *testing.B) {
	var x bitmap.Bitmap
	var y bitmap.Bitmap
	for range 20000 {
		x.Set(uint32(rand.Intn(2000000)))
		y.Set(uint32(rand.Intn(2000000)))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var z bitmap.Bitmap
		z.Or(x)
		z.Or(y)
	}
}
