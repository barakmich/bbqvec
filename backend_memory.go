package bbq

import (
	"errors"
	"math/rand"
	"time"
)

type MemoryBackend struct {
	vecs   []Vector
	rng    *rand.Rand
	dim    int
	nbasis int
}

var _ BuildableBackend = &MemoryBackend{}

func NewMemoryBackend(dimensions int) *MemoryBackend {
	return &MemoryBackend{
		rng: rand.New(rand.NewSource(time.Now().UnixMicro())),
		dim: dimensions,
	}
}

func (mem *MemoryBackend) PutVector(id ID, v Vector) error {
	if len(v) != mem.dim {
		return errors.New("MemoryBackend: vector dimension doesn't match")
	}

	if int(id) < len(mem.vecs) {
		mem.vecs[int(id)] = v
	} else if int(id) == len(mem.vecs) {
		mem.vecs = append(mem.vecs, v)
	} else {
		mem.grow(int(id))
		mem.vecs[int(id)] = v
	}
	return nil
}

func (mem *MemoryBackend) grow(to int) {
	diff := (to - len(mem.vecs)) + 1
	mem.vecs = append(mem.vecs, make([]Vector, diff)...)
}

func (mem *MemoryBackend) ComputeSimilarity(vector Vector, targetID ID) (float32, error) {
	target, err := mem.GetVector(targetID)
	if err != nil {
		return 0, err
	}
	return target.CosineSimilarity(vector), nil
}

func (mem *MemoryBackend) Info() BackendInfo {
	return BackendInfo{
		HasIndexData: false,
		Dimensions:   mem.dim,
		VectorCount:  len(mem.vecs),
	}
}

func (mem *MemoryBackend) GetVector(id ID) (Vector, error) {
	if int(id) > len(mem.vecs)-1 {
		return nil, ErrIDNotFound
	}
	if mem.vecs[int(id)] == nil {
		return nil, ErrIDNotFound
	}
	return mem.vecs[int(id)], nil
}

func (mem *MemoryBackend) GetRandomVector() (Vector, error) {
	n := mem.rng.Intn(len(mem.vecs))
	return mem.vecs[n], nil
}

func (mem *MemoryBackend) ForEachVector(cb func(ID, Vector) error) error {
	for i, v := range mem.vecs {
		err := cb(ID(i), v)
		if err != nil {
			return err
		}
	}
	return nil
}
