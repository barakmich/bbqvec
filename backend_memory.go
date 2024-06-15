package bbq

import (
	"errors"
	"math/rand"
	"time"
)

type MemoryBackend struct {
	vecs []Vector
	rng  *rand.Rand
	dim  int
}

var _ scannableBackend = &MemoryBackend{}
var _ VectorGetter[Vector] = &MemoryBackend{}

func NewMemoryBackend(dimensions int) *MemoryBackend {
	return &MemoryBackend{
		rng: rand.New(rand.NewSource(time.Now().UnixMicro())),
		dim: dimensions,
	}
}

func (mem *MemoryBackend) PutVector(id ID, vector Vector) error {
	if len(vector) != mem.dim {
		return errors.New("MemoryBackend: vector dimension doesn't match")
	}

	if int(id) < len(mem.vecs) {
		mem.vecs[int(id)] = vector
	} else if int(id) == len(mem.vecs) {
		mem.vecs = append(mem.vecs, vector)
	} else {
		mem.grow(int(id))
		mem.vecs[int(id)] = vector
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

func (mem *MemoryBackend) Exists(id ID) bool {
	i := int(id)
	if len(mem.vecs) <= i {
		return false
	}
	return mem.vecs[i] != nil
}

func (mem *MemoryBackend) ForEachVector(cb func(ID) error) error {
	for i, v := range mem.vecs {
		if v == nil {
			continue
		}
		err := cb(ID(i))
		if err != nil {
			return err
		}
	}
	return nil
}
