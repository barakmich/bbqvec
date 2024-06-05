package bbq

import "github.com/RoaringBitmap/roaring"

type VectorBackend interface {
	PutVector(id ID, v Vector) error
	ComputeSimilarity(targetVector Vector, targetID ID) (float32, error)
	Info() BackendInfo
}

type BuildableBackend interface {
	VectorBackend
	GetVector(id ID) (Vector, error)
	GetRandomVector() (Vector, error)
	ForEachVector(func(ID, Vector) error) error
}

type IndexBackend interface {
	SaveBases(bases []Basis) error
	LoadBases() ([]Basis, error)

	SaveBitmap(basis int, index int, bitmap *roaring.Bitmap) error
	LoadBitmap(basis, index int) (*roaring.Bitmap, error)
}

type CompilingBackend interface {
	Compile(logger PrintfFunc) (VectorBackend, error)
}

type BackendInfo struct {
	HasIndexData bool
	Dimensions   int
	NBasis       int
	VectorCount  int
}

func FullTableScanSearch(b BuildableBackend, target Vector, k int) (*ResultSet, error) {
	rs := NewResultSet(k)
	err := b.ForEachVector(func(id ID, v Vector) error {
		sim, err := b.ComputeSimilarity(target, id)
		if err != nil {
			return err
		}
		rs.AddResult(id, sim)
		return nil
	})
	return rs, err
}
