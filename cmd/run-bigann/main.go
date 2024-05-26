package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/alitto/pond"
	"github.com/barakmich/bbq"
)

var (
	path        = flag.String("path", "", "Path to CSVs")
	bases       = flag.Int("bases", 30, "Basis sets")
	spill       = flag.Int("spill", 10, "Spill")
	searchK     = flag.Int("searchk", 20000, "Search K")
	parallelism = flag.Int("parallel", 20, "Parallel queries")
)

func main() {
	flag.Parse()
	if *path == "" {
		log.Fatal("Path is required")
	}
	trainf, err := os.Open(filepath.Join(*path, "train.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer trainf.Close()
	testf, err := os.Open(filepath.Join(*path, "test.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer testf.Close()
	train := loadVecs(trainf)
	test := loadVecs(testf)

	// Now the fun begins
	dim := len(train[0])
	be := bbq.NewMemoryBackend(dim, *bases)
	store, err := bbq.NewVectorStore(be)
	if err != nil {
		log.Fatal(err)
	}

	for i, v := range train {
		store.AddVector(bbq.ID(i), v)
	}

	store.SetLogger(log.Printf)
	start := time.Now()
	store.BuildIndex()
	log.Printf("Built store in %v", time.Since(start))

	neighborf, err := os.Open(filepath.Join(*path, "neighbors.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer neighborf.Close()
	trueres := loadRes(neighborf)

	for i := 0; i < 10; i++ {
		spot := rand.Intn(len(trueres))
		fts, _ := bbq.FullTableScanSearch(be, test[spot], 100)
		ftsrec := fts.ComputeRecall(trueres[spot], 100)
		if ftsrec < 0.999 {
			log.Fatal("Error")
		}
	}
	log.Printf("FTS done")
	res := make([]*bbq.ResultSet, len(test))
	start = time.Now()
	var finished atomic.Uint32
	pool := pond.New(*parallelism, 0, pond.MinWorkers(*parallelism))
	for i, v := range test {
		j, w := i, v
		pool.Submit(func() {
			res[j], err = store.FindNearest(w, 10, *searchK, *spill)
			v := finished.Add(1)
			if v%1000 == 0 {
				log.Printf("Search finished %d", v)
			}
		})
	}
	pool.StopAndWait()
	delta := time.Since(start)
	qps := float64(len(test)) / delta.Seconds()
	totalrecall := 0.0
	for i := range res {
		totalrecall += res[i].ComputeRecall(trueres[i], 10)
	}
	recall := totalrecall / float64(len(res))
	fmt.Println(qps, recall)
}

func loadVecs(f *os.File) []bbq.Vector {
	c := csv.NewReader(f)
	c.ReuseRecord = true
	out := make([]bbq.Vector, 0, 100000)
	for {
		rec, err := c.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		v := make([]float32, len(rec))
		for i, st := range rec {
			x, err := strconv.ParseFloat(st, 32)
			if err != nil {
				log.Fatal(err)
			}
			v[i] = float32(x)
		}
		out = append(out, v)
	}
	return out
}

func loadRes(f *os.File) []*bbq.ResultSet {
	var out []*bbq.ResultSet
	c := csv.NewReader(f)
	c.ReuseRecord = true
	for {
		rec, err := c.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		rs := bbq.NewResultSet(100)
		for i, st := range rec {
			x, err := strconv.Atoi(st)
			if err != nil {
				log.Fatal(err)
			}
			rs.AddResult(bbq.ID(x), float32(150-i))
		}
		out = append(out, rs)
	}
	return out
}
