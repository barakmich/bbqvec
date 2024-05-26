package bbq

import (
	"errors"
	"sync"

	"github.com/kelindar/bitmap"
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
	lefts      []bitmap.Bitmap
	rights     []bitmap.Bitmap
	built      bool
	samples    []Vector
}

func NewVectorStore(backend VectorBackend) (*VectorStore, error) {
	info := backend.Info()
	v := &VectorStore{
		dimensions: info.Dimensions,
		nbasis:     info.NBasis,
		backend:    backend,
		bases:      make([]Basis, info.NBasis),
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

func (vs *VectorStore) FindNearest(vector Vector, k int) (*ResultSet, error) {
	if !vs.built {
		if be, ok := vs.backend.(BuildableBackend); ok {
			return FullTableScanSearch(be, vector, k)
		}
	}
	return vs.findNearestInternal(vector, k)
}

func (vs *VectorStore) findNearestInternal(vector Vector, k int) (*ResultSet, error) {
	bitstring := vs.getBitstring(vector)
	counts := NewCountingBitmap(vs.nbasis)
	// Follow the bitstring and combine/count the bitmaps
	for i := 0; i < vs.nbasis; i++ {
		if bitstring&(0x1<<i) != 0 {
			// Is left
			counts.Or(&vs.lefts[i])
		} else {
			counts.Or(&vs.rights[i])
		}
	}
	// Retrieve the layer with at least K
	elems := counts.TopK(k)
	// Rerank within the reduced set
	rs := NewResultSet(k)
	for _, e := range elems {
		res := vs.backend.ComputeVectorResult(vector, ID(e))
		rs.AddResult(res)
	}
	return rs, nil
}

func (vs *VectorStore) getBitstring(vector Vector) uint64 {
	bitstring := uint64(0)
	for i, basis := range vs.bases {
		last := basis[len(basis)-1]
		final := vs.reduceVector(vector, vs.dimensions-1, basis)
		if vek32.Dot(last, final) > 0 {
			// Is Left
			bitstring |= 0x1 << i
		}
	}
	return bitstring
}

func (vs *VectorStore) BuildIndex() error {
	if vs.built {
		return ErrAlreadyBuilt
	}
	be, ok := vs.backend.(BuildableBackend)
	if !ok {
		return errors.New("Backend does not support building")
	}
	err := vs.makeBasis(be)
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
	for i, bm := range vs.lefts {
		err = be.SaveBitmap(true, i, bm)
		if err != nil {
			return err
		}
	}
	for i, bm := range vs.rights {
		err = be.SaveBitmap(false, i, bm)
		if err != nil {
			return err
		}
	}
	return nil
}

func (vs *VectorStore) makeBasis(be BuildableBackend) error {
	vs.log("Making basis set")
	for n := 0; n < vs.nbasis; n++ {
		i := 0
		basis := make(Basis, 0, vs.dimensions)
		for i < vs.dimensions {
			norm, err := vs.createSplit(i, basis, be)
			if err != nil {
				return err
			}
			basis = append(basis, norm)
			i++
		}
		vs.log("Completed basis %d", n)
		vs.bases[n] = basis
	}
	vs.log("Completed basis set generation")
	return nil
}

func (vs *VectorStore) makeBitmaps(be BuildableBackend) error {
	vs.log("Making bitmaps")
	lefts := make([]bitmap.Bitmap, vs.nbasis)
	rights := make([]bitmap.Bitmap, vs.nbasis)
	var wg sync.WaitGroup
	errs := make([]error, vs.nbasis)
	for n, basis := range vs.bases {
		wg.Add(1)
		go func(n int, basis Basis, wg *sync.WaitGroup) {
			var left bitmap.Bitmap
			var right bitmap.Bitmap
			last := basis[len(basis)-1]
			err := be.ForEachVector(func(i ID, v Vector) error {
				if i != 0 && i%10000 == 0 {
					vs.log("Completed %d for basis %d", i, n)
				}
				final := vs.reduceVector(v, vs.dimensions-1, basis)
				if vek32.Dot(last, final) > 0 {
					left.Set(uint32(i))
				} else {
					right.Set(uint32(i))
				}
				return nil
			})
			if err != nil {
				errs[n] = err
				return
			}
			lefts[n] = left
			rights[n] = right
			vs.log("Finished bitmaps for basis %d. Lefts: %d, Rights: %d", n, left.Count(), right.Count())
			wg.Done()
		}(n, basis, &wg)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	vs.lefts = lefts
	vs.rights = rights
	vs.log("Completed bitmap generation")
	return nil
}

func (vs *VectorStore) reduceVector(vector Vector, depth int, basis Basis) Vector {
	buf := vs.getBuf()
	v := vector.Clone()
	for i := 0; i < depth; i++ {
		v.projectToPlane(basis[i], buf)
	}
	return v
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
	for i := 0; i < vs.nbasis; i++ {
		bm, err := be.LoadBitmap(true, i)
		if err != nil {
			return err
		}
		vs.lefts = append(vs.lefts, bm)
		bm, err = be.LoadBitmap(false, i)
		if err != nil {
			return err
		}
		vs.rights = append(vs.rights, bm)
	}
	vs.built = true
	return nil
}
