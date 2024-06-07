package bbq

import (
	"math"
	"testing"
)

func TestEmpericalCountBitmapConstant(t *testing.T) {
	vecs := NewRandVectorSet(*nVectors, *dim, nil)

	be := NewMemoryBackend(*dim)
	store, err := NewVectorStore(be, *nBasis, 1)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range vecs {
		store.AddVector(ID(i), v)
	}

	store.BuildIndex()

	count := 0
	n := 0
	for _, basisbms := range store.bms {
		for _, bm := range basisbms {
			count += int(bm.GetCardinality())
			n += 1
		}
	}
	t.Logf("Expected avg bitmap count: %0.2f", float64(len(vecs))/float64(2**dim))
	t.Logf("Average bitmap count: %0.2f", float64(count)/float64(n))
	// now we get into the weeds
	buf := make([]float32, store.dimensions)
	maxes := make([]int, 1)
	target := NewRandVector(*dim, nil)
	counts := NewCountingBitmap(*nBasis)
	for i, basis := range store.bases {
		store.findIndexesForBasis(target, basis, buf, maxes)
		for _, m := range maxes {
			if v, ok := store.bms[i][m]; ok {
				counts.Or(v)
			}
		}
		printPredicted(i+1, t)
		t.Logf("got  %#v", counts.cardinalities())
	}
}

const k = 0.83

func printPredicted(i int, t *testing.T) {
	f := make([]float64, i)
	for j := 0; j < i; j++ {
		f[j] = (math.Pow(float64(i), (k*float64(j))+1.0) * float64(*nVectors)) / math.Pow(float64(2**dim), float64(j+1))
	}
	strs := make([]int, i)
	for i, g := range f {
		strs[i] = int(g)
	}
	t.Logf("exp  %#v", strs)
}
