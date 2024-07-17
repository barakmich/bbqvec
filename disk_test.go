package bbq

import (
	"os"
	"testing"
)

func TestDiskBackend(t *testing.T) {
	vecs := NewRandVectorSet(*nVectors, *dim, nil)

	mem := NewMemoryBackend(*dim)

	dir := t.TempDir()
	defer os.RemoveAll(dir)

	be, err := NewDiskBackend(dir, *dim, NoQuantization{})
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewVectorStore(be, *nBasis, 1)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range vecs {
		mem.PutVector(ID(i), v)
		store.AddVector(ID(i), v)
	}
	err = store.Sync()
	if err != nil {
		t.Fatal(err)
	}

	err = store.Close()
	if err != nil {
		t.Fatal(err)
	}

	// Reopen

	be, err = NewDiskBackend(dir, *dim, NoQuantization{})
	if err != nil {
		t.Fatal(err)
	}
	store, err = NewVectorStore(be, *nBasis, 1)
	if err != nil {
		t.Fatal(err)
	}

	targetvecs := NewRandVectorSet(*testvecs, *dim, nil)
	for _, v := range targetvecs {
		fts, err := FullTableScanSearch(mem, v, 20)
		fts.Len()
		if err != nil {
			t.Fatal(err)
		}
	}

}
