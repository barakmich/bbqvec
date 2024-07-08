package bbq

import (
	"errors"
	"math/rand"
	"time"
)

type QuantizedMemoryBackend[V any, Q Quantization[V]] struct {
	vecs         []*V
	rng          *rand.Rand
	dim          int
	quantization Q
}

var _ scannableBackend = &QuantizedMemoryBackend[Vector, NoQuantization]{}
var _ VectorGetter[Vector] = &QuantizedMemoryBackend[Vector, NoQuantization]{}

func NewQuantizedMemoryBackend[V any, Q Quantization[V]](dimensions int, quantization Q) *QuantizedMemoryBackend[V, Q] {
	return &QuantizedMemoryBackend[V, Q]{
		rng:          rand.New(rand.NewSource(time.Now().UnixMicro())),
		dim:          dimensions,
		quantization: quantization,
	}
}

func (q *QuantizedMemoryBackend[V, Q]) Close() error {
	return nil
}

func (q *QuantizedMemoryBackend[V, Q]) PutVector(id ID, vector Vector) error {
	if len(vector) != q.dim {
		return errors.New("QuantizedMemoryBackend: vector dimension doesn't match")
	}

	v, err := q.quantization.Lower(vector)
	if err != nil {
		return err
	}

	if int(id) < len(q.vecs) {
		q.vecs[int(id)] = &v
	} else if int(id) == len(q.vecs) {
		q.vecs = append(q.vecs, &v)
	} else {
		q.grow(int(id))
		q.vecs[int(id)] = &v
	}
	return nil
}

func (q *QuantizedMemoryBackend[V, Q]) grow(to int) {
	diff := (to - len(q.vecs)) + 1
	q.vecs = append(q.vecs, make([]*V, diff)...)
}

func (q *QuantizedMemoryBackend[V, Q]) ComputeSimilarity(vector Vector, targetID ID) (float32, error) {
	v, err := q.quantization.Lower(vector)
	if err != nil {
		return 0, err
	}
	target, err := q.GetVector(targetID)
	if err != nil {
		return 0, err
	}
	return q.quantization.Similarity(target, v), nil
}

func (q *QuantizedMemoryBackend[V, Q]) Info() BackendInfo {
	return BackendInfo{
		HasIndexData: false,
		Dimensions:   q.dim,
	}
}

func (q *QuantizedMemoryBackend[V, Q]) Exists(id ID) bool {
	i := int(id)
	if len(q.vecs) <= i {
		return false
	}
	return q.vecs[i] != nil
}

func (q *QuantizedMemoryBackend[V, Q]) GetVector(id ID) (v V, err error) {
	if int(id) > len(q.vecs)-1 {
		err = ErrIDNotFound
		return
	}
	if q.vecs[int(id)] == nil {
		err = ErrIDNotFound
		return
	}
	return *q.vecs[int(id)], nil
}

func (q *QuantizedMemoryBackend[V, Q]) ForEachVector(cb func(ID) error) error {
	for i, v := range q.vecs {
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
