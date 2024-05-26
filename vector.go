package bbq

import (
	"math/rand"

	"github.com/viterin/vek/vek32"
)

type ID uint64

type Basis []Vector

type Vector []float32

func (v Vector) Clone() Vector {
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

func (v Vector) projectToPlane(normal Vector, buf []float32) {
	dot := vek32.Dot(v, normal)
	vek32.MulNumber_Into(buf, normal, dot)
	vek32.Sub_Inplace(v, buf)
	// normalize in-line
	factor := vek32.Norm(v)
	vek32.DivNumber_Inplace(v, factor)
}

func (v Vector) Normalize() {
	factor := vek32.Norm(v)
	vek32.DivNumber_Inplace(v, factor)
}

func (v Vector) Dimensions() int {
	return len(v)
}

func (v Vector) CosineSimilarity(other Vector) float32 {
	return vek32.CosineSimilarity(v, other)
}

func NewRandVector(dim int, rng *rand.Rand) Vector {
	out := make([]float32, dim)
	for i := 0; i < dim; i++ {
		if rng != nil {
			out[i] = float32(rng.NormFloat64())
		} else {
			out[i] = float32(rand.NormFloat64())
		}
	}
	return out
}
