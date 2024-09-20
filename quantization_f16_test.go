package bbq

import "testing"

func TestFloat16Quantization(t *testing.T) {
	vecs := NewRandVectorSet(1000, *dim, nil)
	mem := NewMemoryBackend(*dim)
	quant := NewQuantizedMemoryBackend(*dim, Float16Quantization{})
	for i, v := range vecs {
		mem.PutVector(ID(i), v)
		quant.PutVector(ID(i), v)
	}
	target := NewRandVector(*dim, nil)
	memrs, err := FullTableScanSearch(mem, target, 20)
	if err != nil {
		t.Fatal(err)
	}
	qrs, err := FullTableScanSearch(quant, target, 20)
	if err != nil {
		t.Fatal(err)
	}
	recall := memrs.ComputeRecall(qrs, 10)
	t.Logf("Recall %0.4f\n", recall)
	t.Logf("\n%s\n%s", memrs, qrs)
}

func TestFloat16Backend(t *testing.T) {
	vecs := NewRandVectorSet(1000, *dim, nil)
	quant := NewQuantizedMemoryBackend(*dim, Float16Quantization{})
	store, err := NewVectorStore(quant, *nBasis, WithPrespill(2))
	if err != nil {
		t.Fatal(err)
	}
	err = store.AddVectorsWithOffset(0, vecs)
	if err != nil {
		t.Fatal(err)
	}

	target := NewRandVector(*dim, nil)
	qrs, err := FullTableScanSearch(quant, target, 20)
	if err != nil {
		t.Fatal(err)
	}
	rs, err := store.FindNearest(target, 20, 20000, 2)
	if err != nil {
		t.Fatal(err)
	}
	recall := rs.ComputeRecall(qrs, 10)
	t.Logf("Recall %0.4f\n", recall)
	t.Logf("\n%s\n%s", rs, qrs)
}
