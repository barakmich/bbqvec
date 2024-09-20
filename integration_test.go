package bbq

import (
	"testing"
)

func TestBasic(t *testing.T) {
	dim := 256
	nBasis := 20
	k := 20
	searchk := 200

	vecs := NewRandVectorSet(100000, dim, nil)

	be := NewMemoryBackend(dim)
	store, err := NewVectorStore(be, nBasis, WithPrespill(2))
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range vecs {
		store.AddVector(ID(i), v)
	}

	store.SetLogger(t.Logf)

	target := NewRandVector(dim, nil)
	indexNearest, err := store.FindNearest(target, k, searchk, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(indexNearest)
	ftsNearest, err := FullTableScanSearch(be, target, k)
	t.Log(ftsNearest)
	recall := indexNearest.ComputeRecall(ftsNearest, k)
	t.Log("Recall: ", recall)
}
