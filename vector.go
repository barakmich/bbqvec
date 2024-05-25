package bbq

import "github.com/viterin/vek/vek32"

type ID uint64

type Basis []Vector

type Vector []float32

func (v Vector) Clone() Vector {
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

func (v Vector) ProjectToPlane(normal Vector, buf []float32) {
}

func (v Vector) Normalize() {
}

func (v Vector) Dimensions() int {
	return len(v)
}

func (v Vector) CosineSimilarity(other Vector) float32 {
	return vek32.CosineSimilarity(v, other)
}
