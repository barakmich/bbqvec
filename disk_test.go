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
	t.Log("TempDir:", dir)
	be, err := NewDiskBackend(dir, *dim, NoQuantization{})
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewVectorStore(be, *nBasis, 1)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range vecs {
		err := mem.PutVector(ID(i), v)
		if err != nil {
			t.Fatal("error mem put", err)
		}
		err = store.AddVector(ID(i), v)
		if err != nil {
			t.Fatal("error store put", err)
		}
		if i%1000 == 0 {
			t.Log("Wrote", i)
		}
	}
	err = store.Sync()
	if err != nil {
		t.Fatal(err)
	}

	err = store.Close()
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Reopening")
	// Reopen

	be, err = NewDiskBackend(dir, *dim, NoQuantization{})
	if err != nil {
		t.Fatal("Couldn't open disk backend", err)
	}
	store, err = NewVectorStore(be, *nBasis, 1)
	if err != nil {
		t.Fatal("Couldn't open vector store", err)
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
