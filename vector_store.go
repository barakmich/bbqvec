package bbq

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/RoaringBitmap/roaring"
	"github.com/viterin/vek/vek32"
)

const defaultMaxSampling = 10000

type PrintfFunc func(string, ...any)

type VectorStore struct {
	logger     PrintfFunc
	backend    VectorBackend
	dimensions int
	nbasis     int
	bases      []Basis
	bms        []map[int]*roaring.Bitmap
	preSpill   int
}

func NewVectorStore(backend VectorBackend, nBasis int, preSpill int) (*VectorStore, error) {
	info := backend.Info()
	if preSpill <= 0 {
		preSpill = 1
	} else if preSpill > info.Dimensions {
		preSpill = info.Dimensions
	}
	v := &VectorStore{
		dimensions: info.Dimensions,
		nbasis:     nBasis,
		backend:    backend,
		bases:      make([]Basis, nBasis),
		bms:        make([]map[int]*roaring.Bitmap, nBasis),
		preSpill:   preSpill,
	}
	if info.HasIndexData {
		err := v.loadFromBackend()
		return v, err
	}
	err := v.makeBasis()
	if err != nil {
		return nil, err
	}
	err = v.Sync()
	if err != nil {
		return nil, err
	}
	return v, nil
}

func (vs *VectorStore) SetLogger(printf PrintfFunc) {
	vs.logger = printf
}

func (vs *VectorStore) log(s string, a ...any) {
	if vs.logger != nil {
		vs.logger(s, a...)
	}
}

func (vs *VectorStore) AddVector(id ID, v Vector) error {
	if vs.backend.Exists(id) {
		vs.removeFromBitmaps(id)
	}
	err := vs.backend.PutVector(id, v)
	if err != nil {
		return err
	}
	return vs.addToBitmaps([]ID{id}, []Vector{v})
}

func (vs *VectorStore) AddVectorsWithOffset(offset ID, vecs []Vector) error {
	ids := make([]ID, len(vecs))
	for i, v := range vecs {
		id := offset + ID(i)
		ids[i] = id
		if vs.backend.Exists(id) {
			vs.removeFromBitmaps(id)
		}
		vs.backend.PutVector(id, v)
	}
	return vs.addToBitmaps(ids, vecs)
}

func (vs *VectorStore) AddVectorsWithIDs(ids []ID, vecs []Vector) error {
	for i, v := range vecs {
		id := ids[i]
		if vs.backend.Exists(id) {
			vs.removeFromBitmaps(id)
		}
		vs.backend.PutVector(id, v)
	}
	return vs.addToBitmaps(ids, vecs)
}

func (vs *VectorStore) FindNearest(vector Vector, k int, searchk int, spill int) (*ResultSet, error) {
	if spill < 0 {
		spill = 0
	} else if spill >= vs.dimensions {
		spill = vs.dimensions - 1
	}
	return vs.findNearestInternal(vector, k, searchk, spill)
}

func (vs *VectorStore) findNearestInternal(vector Vector, k int, searchk int, spill int) (*ResultSet, error) {
	counts := NewCountingBitmap(vs.nbasis)
	buf := make([]float32, vs.dimensions)
	maxes := make([]int, spill+1)
	for i, basis := range vs.bases {
		spillClone := roaring.New()
		vs.findIndexesForBasis(vector, basis, buf, maxes)
		for _, m := range maxes {
			if v, ok := vs.bms[i][m]; ok {
				spillClone.Or(v)
			}
		}
		counts.Or(spillClone)
	}
	elems := counts.TopK(searchk)
	//vs.log("Actual searchK is: %s", counts.String())
	// Rerank within the reduced set
	rs := NewResultSet(k)
	var err error

	elems.Iterate(func(x uint32) bool {
		// things that take closures should really return error, so that it can abort...
		var sim float32
		sim, err = vs.backend.ComputeSimilarity(vector, ID(x))
		if err != nil {
			return false
		}
		// On err, this will be the zero value of sum (but that's ok, we're going down)
		rs.AddResult(ID(x), sim)
		return true
	})
	return rs, err
}

func (vs *VectorStore) findIndexesForBasis(target Vector, basis Basis, buf []float32, maxes []int) {
	for x, b := range basis {
		dot := vek32.Dot(b, target)
		buf[x] = dot
	}
	for i := 0; i < len(maxes); i++ {
		big := vek32.ArgMax(buf)
		small := vek32.ArgMin(buf)
		idx := 0
		if math.Abs(float64(buf[big])) >= math.Abs(float64(buf[small])) {
			idx = big
		} else {
			idx = small
		}
		if buf[idx] > 0.0 {
			maxes[i] = idx + 1
		} else {
			maxes[i] = -(idx + 1)
		}
		buf[idx] = 0.0
	}
}

func (vs *VectorStore) BuildIndex() error {
	err := vs.Sync()
	if err != nil {
		return err
	}

	if be, ok := vs.backend.(CompilingBackend); ok {
		vs.log("Compiling backend")
		newbe, err := be.Compile(vs.log)
		if err != nil {
			return err
		}
		if newbe != nil {
			vs.backend = newbe
		}
		vs.log("Completed compilation")
	}
	return nil
}

func (vs *VectorStore) Sync() error {
	be, ok := vs.backend.(IndexBackend)
	if !ok {
		return nil
	}
	err := be.SaveBases(vs.bases)
	if err != nil {
		return err
	}
	for b, dimmap := range vs.bms {
		for i, v := range dimmap {
			err = be.SaveBitmap(b, i, v)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (vs *VectorStore) makeBasis() error {
	vs.log("Making basis set")
	for n := 0; n < vs.nbasis; n++ {
		basis := make(Basis, vs.dimensions)
		for i := range vs.dimensions {
			basis[i] = NewRandVector(vs.dimensions, nil)
		}
		for range 10 {
			orthonormalize(basis)
		}
		vs.log("Completed basis %d", n)
		vs.bases[n] = basis
	}
	vs.log("Completed basis set generation")
	return nil
}

// Use Modified Gram-Schmidt (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
// to turn our random vectors into an orthonormal basis.
func orthonormalize(basis Basis) {
	buf := make([]float32, len(basis[0]))
	cur := basis[0]
	for i := 1; i < len(basis); i++ {
		for j := i; j < len(basis); j++ {
			dot := vek32.Dot(basis[j], cur)
			vek32.MulNumber_Into(buf, cur, dot)
			vek32.Sub_Inplace(basis[j], buf)
			basis[j].Normalize()
		}
		cur = basis[i]
	}
}

func debugPrintBasis(basis Basis) {
	for i := 0; i < len(basis); i++ {
		sim := make([]any, len(basis))
		for j := 0; j < len(basis); j++ {
			sim[j] = vek32.CosineSimilarity(basis[i], basis[j])
		}
		pattern := strings.Repeat("%+.15f  ", len(basis))
		fmt.Printf(pattern+"\n", sim...)
	}
}

func (vs *VectorStore) removeFromBitmaps(id ID) {
	for _, m := range vs.bms {
		if m == nil {
			continue
		}
		for _, bm := range m {
			bm.Remove(uint32(id))
		}
	}
}

func (vs *VectorStore) addToBitmaps(ids []ID, vectors []Vector) error {
	var wg sync.WaitGroup
	for n, basis := range vs.bases {
		wg.Add(1)
		go func(n int, basis Basis, wg *sync.WaitGroup) {
			if v := vs.bms[n]; v == nil {
				vs.bms[n] = make(map[int]*roaring.Bitmap)
			}
			buf := make([]float32, vs.dimensions)
			maxes := make([]int, vs.preSpill)
			for i := range vectors {
				id := ids[i]
				v := vectors[i]
				vs.findIndexesForBasis(v, basis, buf, maxes)
				for _, m := range maxes {
					if _, ok := vs.bms[n][m]; !ok {
						vs.bms[n][m] = roaring.NewBitmap()
					}
					vs.bms[n][m].Add(uint32(id))
				}
			}
			wg.Done()
		}(n, basis, &wg)
	}
	wg.Wait()
	return nil
}

func (vs *VectorStore) loadFromBackend() error {
	var err error
	be, ok := vs.backend.(IndexBackend)
	if !ok {
		panic("Backend suggested that it could store indexes, but it cannot")
	}
	vs.bases, err = be.LoadBases()
	if err != nil {
		return err
	}
	for b := 0; b < vs.nbasis; b++ {
		dimmap := make(map[int]*roaring.Bitmap)
		for i := 1; i <= vs.dimensions; i++ {
			bm, err := be.LoadBitmap(b, i)
			if err != nil {
				return err
			}
			dimmap[i] = bm
			bm, err = be.LoadBitmap(b, -i)
			if err != nil {
				return err
			}
			dimmap[-i] = bm
		}
	}
	return nil
}
