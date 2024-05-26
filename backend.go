package bbq

import "github.com/kelindar/bitmap"

type VectorBackend interface {
	PutVector(id ID, v Vector) error
	ComputeVectorResult(vector Vector, targetID ID) *Result
	Info() BackendInfo
}

type BuildableBackend interface {
	VectorBackend
	GetRandomVector() (Vector, error)
	ForEachVector(func(ID, Vector) error) error
}

type IndexBackend interface {
	SaveBases(bases []Basis) error
	LoadBases() ([]Basis, error)

	SaveBitmap(isLeft bool, index int, bitmap bitmap.Bitmap) error
	LoadBitmap(isLeft bool, index int) (bitmap.Bitmap, error)
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
		sim := v.CosineSimilarity(target)
		rs.AddResult(&Result{
			ID:         id,
			Similarity: sim,
			Vector:     v,
		})
		return nil
	})
	return rs, err
}
