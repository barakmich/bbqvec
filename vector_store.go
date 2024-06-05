package bbq

import (
	"errors"
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
	built      bool
}

func NewVectorStore(backend VectorBackend) (*VectorStore, error) {
	info := backend.Info()
	v := &VectorStore{
		dimensions: info.Dimensions,
		nbasis:     info.NBasis,
		backend:    backend,
		bases:      make([]Basis, info.NBasis),
		bms:        make([]map[int]*roaring.Bitmap, info.NBasis),
	}
	if info.HasIndexData {
		err := v.loadFromBackend()
		return v, err
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
	if vs.built {
		// TODO: Add to bitmaps, save the bitmaps periodically, add to backing store
		return ErrAlreadyBuilt
	}
	return vs.backend.PutVector(id, v)
}

func (vs *VectorStore) FindNearest(vector Vector, k int, searchk int, spill int) (*ResultSet, error) {
	if !vs.built {
		if be, ok := vs.backend.(BuildableBackend); ok {
			return FullTableScanSearch(be, vector, k)
		}
	}
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
		for x, b := range basis {
			dot := vek32.Dot(b, vector)
			buf[x] = dot
		}
		for i := 0; i < spill+1; i++ {
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

func (vs *VectorStore) BuildIndex() error {
	if vs.built {
		return ErrAlreadyBuilt
	}
	be, ok := vs.backend.(BuildableBackend)
	if !ok {
		return errors.New("Backend does not support building")
	}
	err := vs.makeBasis()
	if err != nil {
		return err
	}

	err = vs.makeBitmaps(be)
	if err != nil {
		return err
	}

	err = vs.saveAll()
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
	vs.built = true
	vs.log("Index complete")
	return err
}

func (vs *VectorStore) saveAll() error {
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

func printBasis(basis Basis) {
	for i := 0; i < len(basis); i++ {
		sim := make([]any, len(basis))
		for j := 0; j < len(basis); j++ {
			sim[j] = vek32.CosineSimilarity(basis[i], basis[j])
		}
		pattern := strings.Repeat("%+.15f  ", len(basis))
		fmt.Printf(pattern+"\n", sim...)
	}
}

func (vs *VectorStore) makeBitmaps(be BuildableBackend) error {
	vs.log("Making bitmaps")
	var wg sync.WaitGroup
	errs := make([]error, vs.nbasis)
	for n, basis := range vs.bases {
		wg.Add(1)
		go func(n int, basis Basis, wg *sync.WaitGroup) {
			out := make(map[int]*roaring.Bitmap)
			buf := make([]float32, vs.dimensions)
			be.ForEachVector(func(i ID, v Vector) error {
				if i != 0 && i%10000 == 0 {
					vs.log("Completed %d of basis %d", i, n)
				}
				for x, b := range basis {
					dot := vek32.Dot(b, v)
					buf[x] = dot
				}
				big := vek32.ArgMax(buf)
				small := vek32.ArgMin(buf)
				idx := 0
				trueMax := 0
				if math.Abs(float64(buf[big])) >= math.Abs(float64(buf[small])) {
					idx = big
				} else {
					idx = small
				}
				if buf[idx] > 0.0 {
					trueMax = idx + 1
				} else {
					trueMax = -(idx + 1)
				}

				if _, ok := out[trueMax]; !ok {
					out[trueMax] = roaring.NewBitmap()
				}
				out[trueMax].Add(uint32(i))
				return nil
			})
			vs.bms[n] = out
			vs.log("Finished bitmaps for basis %d. Approx %d per face", n, out[1].GetCardinality())
			wg.Done()
		}(n, basis, &wg)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	vs.log("Completed bitmap generation")
	return nil
}

func (vs *VectorStore) getBuf() []float32 {
	// TODO: create a sync.pool?
	return make([]float32, vs.dimensions)
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
	vs.built = true
	return nil
}
