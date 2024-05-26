package bbq

import (
	"math/rand"
	"testing"
)

func buildVectors(n int, dim int, rng *rand.Rand) []Vector {
	out := make([]Vector, n)
	for i := 0; i < n; i++ {
		out[i] = NewRandVector(dim, rng)
		out[i].Normalize()
	}
	return out
}

func TestBasic(t *testing.T) {
	dim := 256
	nBasis := 10
	k := 20

	vecs := buildVectors(100000, dim, nil)

	be := NewMemoryBackend(dim, nBasis)
	store, err := NewVectorStore(be)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range vecs {
		store.AddVector(ID(i), v)
	}

	store.SetLogger(t.Logf)
	store.BuildIndex()

	target := NewRandVector(dim, nil)
	indexNearest, err := store.FindNearest(target, k)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(indexNearest)
	ftsNearest, err := FullTableScanSearch(be, target, k)
	t.Log(ftsNearest)
	recall := indexNearest.ComputeRecall(ftsNearest)
	if recall < 0.999 {
		t.Fatal("imperfect recall", recall)
	}
}
