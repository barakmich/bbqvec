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
	"runtime/pprof"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	bbq "github.com/daxe-ai/bbqvec"
)

var (
	k           = flag.Int("k", 10, "K top results")
	path        = flag.String("path", "", "Path to CSVs")
	bases       = flag.Int("bases", 20, "Basis sets")
	spill       = flag.Int("spill", 10, "Spill")
	searchK     = flag.Int("searchk", 10000, "Search K")
	parallelism = flag.Int("parallel", 20, "Parallel queries")
	cpuprofile  = flag.String("cpuprof", "", "CPU Profile file")
)

func main() {
	flag.Parse()
	if *path == "" {
		log.Fatal("Path is required")
	}
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close() // error handling omitted for example
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
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
	log.Println("Loading Train")
	train := loadVecs(trainf)
	log.Println("Train has", len(train))
	log.Println("Loading Test")
	test := loadVecs(testf)
	log.Println("Test has", len(test))

	log.Println("Loading true neighbors")

	neighborf, err := os.Open(filepath.Join(*path, "neighbors.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer neighborf.Close()
	trueres := loadRes(neighborf)

	// Now the fun begins
	dim := len(train[0])
	log.Println("Loading into memory")
	be := bbq.NewMemoryBackend(dim)
	store, err := bbq.NewVectorStore(be, *bases, 1)
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	for i, v := range train {
		store.AddVector(bbq.ID(i), v)
		if i%1000 == 0 {
			log.Printf(".")
		}
	}
	log.Printf("\n")
	store.SetLogger(log.Printf)
	log.Printf("Built store in %v", time.Since(start))

	for i := 0; i < 10; i++ {
		spot := rand.Intn(len(trueres))
		fts, _ := bbq.FullTableScanSearch(be, test[spot], 100)
		ftsrec := fts.ComputeRecall(trueres[spot], 100)
		if ftsrec < 0.98 {
			log.Fatal("Error")
		}
	}
	log.Printf("FullTableScan data spot check done")
	res := make([]*bbq.ResultSet, len(test))
	var finished atomic.Uint32
	var wg sync.WaitGroup
	ch := make(chan pair)
	for i := 0; i < *parallelism; i++ {
		go func() {
			for p := range ch {
				res[p.id], err = store.FindNearest(p.vec, *k, *searchK, *spill)
				v := finished.Add(1)
				if v%1000 == 0 {
					log.Printf("Search finished %d", v)
				}
			}
			wg.Done()
		}()
		wg.Add(1)
	}
	start = time.Now()
	for i, v := range test {
		ch <- pair{i, v}
	}
	close(ch)
	wg.Wait()
	delta := time.Since(start)
	qps := float64(len(test)) / delta.Seconds()
	totalrecall := 0.0
	for i := range res {
		totalrecall += res[i].ComputeRecall(trueres[i], 10)
	}
	recall := totalrecall / float64(len(res))
	fmt.Printf("%0.4f,%0.4f", qps, recall)
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

type pair struct {
	id  int
	vec bbq.Vector
}
