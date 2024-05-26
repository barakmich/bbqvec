package bbq

import (
	"errors"
	"math/rand"
	"time"
)

type MemoryBackend struct {
	vecs   []Vector
	ids    map[ID]int
	rng    *rand.Rand
	dim    int
	nbasis int
}

var _ BuildableBackend = &MemoryBackend{}

func NewMemoryBackend(dimensions int, nBasis int) *MemoryBackend {
	return &MemoryBackend{
		ids:    make(map[ID]int),
		rng:    rand.New(rand.NewSource(time.Now().UnixMicro())),
		nbasis: nBasis,
		dim:    dimensions,
	}
}

func (mem *MemoryBackend) PutVector(id ID, v Vector) error {
	if len(v) != mem.dim {
		return errors.New("MemoryBackend: vector dimension doesn't match")
	}
	mem.ids[id] = len(mem.vecs)
	mem.vecs = append(mem.vecs, v)
	return nil
}

func (mem *MemoryBackend) ComputeSimilarity(vector Vector, targetID ID) float32 {
	target := mem.vecs[targetID]
	return target.CosineSimilarity(vector)
}

func (mem *MemoryBackend) Info() BackendInfo {
	return BackendInfo{
		HasIndexData: false,
		Dimensions:   mem.dim,
		NBasis:       mem.nbasis,
		VectorCount:  len(mem.vecs),
	}
}

func (mem *MemoryBackend) GetVector(id ID) (Vector, error) {
	return mem.vecs[id], nil
}

func (mem *MemoryBackend) GetRandomVector() (Vector, error) {
	n := mem.rng.Intn(len(mem.vecs))
	return mem.vecs[n], nil
}

func (mem *MemoryBackend) ForEachVector(cb func(ID, Vector) error) error {
	for k, v := range mem.ids {
		err := cb(k, mem.vecs[v])
		if err != nil {
			return err
		}
	}
	return nil
}
