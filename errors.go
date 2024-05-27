package bbq

import "errors"

var (
	ErrAlreadyBuilt = errors.New("Already built the index")
	ErrIDNotFound   = errors.New("ID not found")
)
