package bbq

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/RoaringBitmap/roaring"
	"github.com/barakmich/mmap-go"
)

const defaultVecsPerFile = 200000

type DiskBackend[V any] struct {
	dir          string
	metadata     diskMetadata
	quantization Quantization[V]
	vectorPages  map[int]mmap.MMap
	vectorFiles  map[int]*os.File
	token        uint64
}

type diskMetadata struct {
	Dimensions   int    `json:"dimensions"`
	Quantization string `json:"quantization"`
	VecsPerFile  int    `json:"vecs_per_file"`
	VecFiles     []int  `json:"vec_files"`
}

var _ IndexBackend = &DiskBackend[Vector]{}

func NewDiskBackend[V any](directory string, dimensions int, quantization Quantization[V]) (*DiskBackend[V], error) {
	token := rand.Uint64()
	if token == 0 {
		token = 1
	}
	be := &DiskBackend[V]{
		dir: directory,
		metadata: diskMetadata{
			Dimensions:   dimensions,
			Quantization: quantization.Name(),
			VecsPerFile:  defaultVecsPerFile,
		},
		quantization: quantization,
		token:        token,
		vectorPages:  make(map[int]mmap.MMap),
		vectorFiles:  make(map[int]*os.File),
	}
	err := be.openFiles()
	if err != nil {
		return nil, err
	}
	return be, nil
}

func (d *DiskBackend[V]) Close() error {
	err := d.Sync()
	if err != nil {
		return err
	}
	for _, mm := range d.vectorPages {
		err := mm.Unmap()
		if err != nil {
			return err
		}
	}
	for _, f := range d.vectorFiles {
		err := f.Close()
		if err != nil {
			return err
		}
	}
	return d.saveMetadata()
}

func (d *DiskBackend[V]) Sync() error {
	for _, mm := range d.vectorPages {
		err := mm.FlushAsync()
		if err != nil {
			return err
		}
	}
	return nil
}

func (d *DiskBackend[V]) openFiles() error {
	_, err := os.Stat(d.dir)
	if errors.Is(err, fs.ErrNotExist) {
		return d.createNew()
	} else if err != nil {
		return err
	}

	_, err = os.Stat(filepath.Join(d.dir, "metadata.json"))
	if errors.Is(err, fs.ErrNotExist) {
		return d.createNew()
	} else if err != nil {
		return err
	}

	f, err := os.Open(filepath.Join(d.dir, "metadata.json"))
	if err != nil {
		return err
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(&d.metadata)
	if err != nil {
		return err
	}

	for _, k := range d.metadata.VecFiles {
		f, err := os.OpenFile(mkPageFilepath(d.dir, k), os.O_RDWR, 0755)
		if err != nil {
			return err
		}
		d.vectorFiles[k] = f
		mm, err := mmap.Map(f, mmap.RDWR, 0)
		if err != nil {
			return err
		}
		d.vectorPages[k] = mm
	}
	return nil
}

func (d *DiskBackend[V]) createNew() error {
	err := os.MkdirAll(d.dir, 0755)
	if err != nil {
		return err
	}
	return d.saveMetadata()
}

func (d *DiskBackend[V]) saveMetadata() error {
	f, err := os.Create(filepath.Join(d.dir, "metadata.json"))
	if err != nil {
		return err
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(d.metadata)
	if err != nil {
		return err
	}
	return nil
}

func (d *DiskBackend[V]) PutVector(id ID, v Vector) error {
	var err error
	key := int(id) / d.metadata.VecsPerFile
	off := int(id) % d.metadata.VecsPerFile
	page, ok := d.vectorPages[key]
	if !ok {
		page, err = d.createPage(key)
		if err != nil {
			return err
		}
	}
	size := d.quantization.LowerSize(d.metadata.Dimensions)
	l, err := d.quantization.Lower(v)
	if err != nil {
		return err
	}
	slice := page[off*size : (off+1)*size]
	return d.quantization.Marshal(slice, l)
}

func (d *DiskBackend[V]) createPage(key int) (mmap.MMap, error) {
	f, err := os.Create(mkPageFilepath(d.dir, key))
	if err != nil {
		return nil, err
	}
	vecsize := d.quantization.LowerSize(d.metadata.Dimensions)
	err = f.Truncate(int64(vecsize * d.metadata.VecsPerFile))
	if err != nil {
		return nil, err
	}
	d.vectorFiles[key] = f
	mm, err := mmap.Map(f, mmap.RDWR, 0)
	if err != nil {
		return nil, err
	}
	d.vectorPages[key] = mm
	d.metadata.VecFiles = append(d.metadata.VecFiles, key)
	err = d.saveMetadata()
	if err != nil {
		return nil, err
	}
	return mm, nil
}

func (d *DiskBackend[V]) ComputeSimilarity(targetVector Vector, targetID ID) (float32, error) {
	v, err := d.quantization.Lower(targetVector)
	if err != nil {
		return 0, err
	}
	target, err := d.GetVector(targetID)
	if err != nil {
		return 0, err
	}
	return d.quantization.Similarity(target, v), nil
}

func (d *DiskBackend[V]) Info() BackendInfo {
	exists := true
	if _, err := os.Stat(filepath.Join(d.dir, "bases")); errors.Is(err, os.ErrNotExist) {
		exists = false
	}

	return BackendInfo{
		HasIndexData: exists,
		Dimensions:   d.metadata.Dimensions,
		Quantization: d.quantization.Name(),
	}
}

func (d *DiskBackend[V]) Exists(id ID) bool {
	key := int(id) / d.metadata.VecsPerFile
	off := int(id) % d.metadata.VecsPerFile
	page, ok := d.vectorPages[key]
	if !ok {
		return false
	}
	size := d.quantization.LowerSize(d.metadata.Dimensions)
	slice := page[off*size : (off+1)*size]
	for _, x := range slice {
		if x != 0x0 {
			return true
		}
	}
	return false
}

func (d *DiskBackend[V]) GetVector(id ID) (v V, err error) {
	key := int(id) / d.metadata.VecsPerFile
	off := int(id) % d.metadata.VecsPerFile
	page, ok := d.vectorPages[key]
	if !ok {
		err = ErrIDNotFound
		return
	}
	size := d.quantization.LowerSize(d.metadata.Dimensions)
	slice := page[off*size : (off+1)*size]
	return d.quantization.Unmarshal(slice)
}

func (d *DiskBackend[V]) SaveBases(bases []Basis, token uint64) (uint64, error) {
	if token == d.token {
		return d.token, nil
	}
	nbuf := make([]byte, 4)
	buf := bytes.NewBuffer(nil)
	for _, b := range bases {
		for _, v := range b {
			for _, s := range v {
				binary.LittleEndian.PutUint32(nbuf, math.Float32bits(s))
				buf.Write(nbuf)
			}
		}
	}
	f, err := os.Create(filepath.Join(d.dir, "bases"))
	if err != nil {
		return 0, err
	}
	defer f.Close()
	_, err = io.Copy(f, buf)
	return d.token, err
}

func (d *DiskBackend[V]) LoadBases() ([]Basis, error) {
	f, err := os.Open(filepath.Join(d.dir, "bases"))
	if err != nil {
		return nil, err
	}
	var out []Basis
	var basis Basis
	var vec Vector
	buf := make([]byte, 4)
	for {
		_, err = f.Read(buf)
		if errors.Is(err, io.EOF) {
			break
		}
		entry := math.Float32frombits(binary.LittleEndian.Uint32(buf))
		vec = append(vec, entry)
		if len(vec) == d.metadata.Dimensions {
			basis = append(basis, vec)
			vec = nil
			if len(basis) == d.metadata.Dimensions {
				out = append(out, basis)
				basis = nil
			}
		}
	}
	return out, nil
}

func (d *DiskBackend[V]) SaveBitmap(basis int, index int, bitmap *roaring.Bitmap) error {
	path := mkBmapFilepath(d.dir, basis, index)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = bitmap.WriteTo(f)
	return err
}

func (d *DiskBackend[V]) LoadBitmap(basis int, index int) (*roaring.Bitmap, error) {
	f, err := os.Open(mkBmapFilepath(d.dir, basis, index))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	bm := roaring.NewBitmap()
	_, err = bm.ReadFrom(f)
	return bm, err
}

func mkPageFilepath(basedir string, key int) string {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(key))
	indexStr := hex.EncodeToString(buf)
	return filepath.Join(basedir, fmt.Sprintf("%s.vec", indexStr))
}

func mkBmapFilepath(basedir string, basis int, index int) string {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint16(buf, uint16(basis))
	basisStr := hex.EncodeToString(buf[:2])
	binary.BigEndian.PutUint32(buf, uint32(index))
	indexStr := hex.EncodeToString(buf[:4])
	return filepath.Join(basedir, fmt.Sprintf("%s-%s.bmap", basisStr, indexStr))
}
