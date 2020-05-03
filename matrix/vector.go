package matrix

import (
	"errors"
	"fmt"
	"math"
)

// Vector represents a vector of values
type Vector struct {
	Values []float32
}

// InitializeVector an empty matrix of the specified size
func InitializeVector(size int) (*Vector, error) {
	values := make([]float32, size)
	return &Vector{Values: values}, nil
}

// String representation of the Vector
func (v Vector) String() string {
	result := ""
	for i := 0; i < len(v.Values); i++ {
		result += fmt.Sprintf("| %v ", v.Values[i])
	}
	result += "|"
	return result
}

// Size returns length of vector
func (v Vector) Size() int {
	return len(v.Values)
}

// EuclideanDistance calculates euclidean distance between two vectors
func (v *Vector) EuclideanDistance(other *Vector) (float32, error) {
	var err error
	if v.Size() != other.Size() {
		return -1, errors.New("Vector sizes do not match")
	}
	secondNeg, err := other.MultiplyConstant(-1)
	diff, err := v.Add(secondNeg)
	squared, err := diff.ApplyFunc(func(n float32) float32 {
		return n * n
	})
	return float32(math.Sqrt(float64(squared.Sum()))), err
}

// Sum all values in vector
func (v *Vector) Sum() float32 {
	var sum float32
	for _, elem := range v.Values {
		sum += elem
	}
	return sum
}

// Add add constant to all elements of vector
func (v *Vector) Add(other *Vector) (*Vector, error) {
	var err error
	var result *Vector
	if v.Size() != other.Size() {
		return result, errors.New("Vector sizes do not match")
	}
	result, err = InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] + other.Values[i]
	}
	return result, err
}

// AddConstant add constant to all elements of vector
func (v *Vector) AddConstant(constant float32) (*Vector, error) {
	result, err := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] + constant
	}
	return result, err
}

// MultiplyConstant multiply constant to all elements of matrix
func (v *Vector) MultiplyConstant(constant float32) (*Vector, error) {
	result, err := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] * constant
	}
	return result, err
}

// ApplyFunc apply function to all elements of vector
func (v *Vector) ApplyFunc(applier applier) (*Vector, error) {
	result, err := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = applier(v.Values[i])
	}
	return result, err
}
