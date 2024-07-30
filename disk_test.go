package bbq

import (
	"testing"
)

func TestDiskBackend(t *testing.T) {
	testDiskBackendQuantization(t, NoQuantization{})
}

func TestDiskBackendF16(t *testing.T) {
	testDiskBackendQuantization(t, Float16Quantization{})
}

func testDiskBackendQuantization[L any](t *testing.T, q Quantization[L]) {
	vecs := NewRandVectorSet(*nVectors, *dim, nil)

	mem := NewMemoryBackend(*dim)

	dir := t.TempDir()
	t.Log("TempDir:", dir)
	be, err := NewDiskBackend(dir, *dim, q)
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

	be, err = NewDiskBackend(dir, *dim, q)
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
