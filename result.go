package bbq

import (
	"fmt"
	"sync"
)

type Result struct {
	Similarity float32
	ID         ID
	Vector     Vector
}

func (r Result) String() string {
	return fmt.Sprintf("(%d %0.4f)", r.ID, r.Similarity)
}

type ResultSet struct {
	inner sync.Mutex
	set   []*Result
	k     int
}

func NewResultSet(topK int) *ResultSet {
	return &ResultSet{
		k: topK,
	}
}

func (rs *ResultSet) Len() int {
	return len(rs.set)
}

func (rs *ResultSet) UnionSet(other *ResultSet) *ResultSet {
	if rs == nil {
		return other
	}
	newK := rs.k + other.k
	outSet := make([]*Result, 0, newK)
	lx := 0
	rx := 0
	for {
		if len(rs.set) == 0 {
			outSet = append(outSet, other.set[rx:]...)
			break
		}
		if len(other.set) == 0 {
			outSet = append(outSet, rs.set[lx:]...)
			break
		}
		topl := rs.set[lx]
		topr := other.set[rx]
		if topl.ID == topr.ID {
			outSet = append(outSet, topl)
			lx++
			rx++
		}
		if topl.Similarity >= topr.Similarity {
			outSet = append(outSet, topl)
			lx++
		} else {
			outSet = append(outSet, topr)
			rx++
		}
	}
	return &ResultSet{
		set: outSet,
		k:   newK,
	}
}

func (rs *ResultSet) MergeSet(other *ResultSet) *ResultSet {
	n := rs.UnionSet(other)

	targetK := other.k
	if other.k < rs.k {
		targetK = rs.k
	}
	n.k = targetK
	n.set = n.set[:min(len(n.set), n.k)]
	return n
}

func (rs *ResultSet) ExtendResults(other []*Result) {
	for _, v := range other {
		rs.AddResult(v)
	}
}

func (rs *ResultSet) ComputeRecall(baseline *ResultSet) float32 {
	found := 0
	for _, v := range baseline.set {
		for _, w := range rs.set {
			if v.ID == w.ID {
				found += 1
			}
		}
	}
	return float32(found) / float32(baseline.k)
}

func (rs *ResultSet) String() string {
	return fmt.Sprint(rs.set)
}

func (rs *ResultSet) AddResult(result *Result) bool {
	// Do a quick check...
	if len(rs.set) == rs.k {
		// Bail if the last one beats us
		last := rs.set[len(rs.set)-1]
		if last.Similarity > result.Similarity {
			return false
		}
	}
	for _, v := range rs.set {
		if v.ID == result.ID {
			return true
		}
	}
	rs.inner.Lock()
	defer rs.inner.Unlock()
	insert := 0
	found := false
	for insert != rs.k {
		// If we're building it out, then the new insertion point is at the end.
		if len(rs.set) <= insert {
			found = true
			break
		}
		if rs.set[insert].Similarity < result.Similarity {
			found = true
			break
		}
		insert++
	}
	if !found {
		return false
	}
	rs.set = append(rs.set[:insert], append([]*Result{result}, rs.set[insert:]...)...)
	rs.set = rs.set[:min(len(rs.set), rs.k)]
	return true
}

func (rs *ResultSet) ToSlice() []*Result {
	return rs.set
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
