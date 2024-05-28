package bbq

import (
	"github.com/viterin/vek/vek32"
)

const vectorsToConsider = 200

func (vs *VectorStore) getRandomVec(depth int, basis Basis, be BuildableBackend) (Vector, error) {
	r, err := be.GetRandomVector()
	if err != nil {
		return nil, err
	}
	return vs.reduceVector(r, depth, basis), nil
}

// This function is largely modeled off of [Annoy](https://github.com/spotify/annoy) -- we want to do the same trick and find a splitting hyperplane
func (vs *VectorStore) createSplit(depth int, currentBasis Basis, be BuildableBackend) (Vector, error) {
	// First, find a random vector in the set
	p, err := vs.getRandomVec(depth, currentBasis, be)
	if err != nil {
		return nil, err
	}

	// Now, find something that's "far" from p
	var q Vector
	bestdist := float32(-2.0)

	for i := 0; i < vectorsToConsider; i++ {
		candidate, err := vs.getRandomVec(depth, currentBasis, be)
		if err != nil {
			return nil, err
		}
		dist := vek32.Distance(candidate, p)
		if dist > bestdist {
			q = candidate
			bestdist = dist
		}
	}

	p.Normalize()
	q.Normalize()
	vek32.Sub_Inplace(p, q)
	p.Normalize()
	return p, nil
}
