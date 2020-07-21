package matrix

import (
	"errors"
	"fmt"
	"math"
)

// Vector represents a vector of values
type Vector struct {
	Values []float64
}

// InitializeVector an empty matrix of the specified size
func InitializeVector(size int) *Vector {
	values := make([]float64, size)
	return &Vector{Values: values}
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
func (v *Vector) EuclideanDistance(other *Vector) (float64, error) {
	var err error
	if v.Size() != other.Size() {
		return -1, errors.New("Vector sizes do not match")
	}
	secondNeg := other.MultiplyConstant(-1)
	diff, err := v.Add(secondNeg)
	squared := diff.ApplyFunc(func(n float64) float64 {
		return n * n
	})
	return float64(math.Sqrt(float64(squared.Sum()))), err
}

// Sum all values in vector
func (v *Vector) Sum() (sum float64) {
	sum = 0.0
	for _, elem := range v.Values {
		sum += elem
	}
	return
}

// Add add constant to all elements of vector
func (v *Vector) Add(other *Vector) (result *Vector, err error) {
	if v.Size() != other.Size() {
		return result, errors.New("Vector sizes do not match")
	}
	result = InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] + other.Values[i]
	}
	return result, err
}

// AddConstant add constant to all elements of vector
func (v *Vector) AddConstant(constant float64) *Vector {
	result := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] + constant
	}
	return result
}

// MultiplyConstant multiply constant to all elements of matrix
func (v *Vector) MultiplyConstant(constant float64) *Vector {
	result := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = v.Values[i] * constant
	}
	return result
}

// ApplyFunc apply function to all elements of vector
func (v *Vector) ApplyFunc(applier applier) *Vector {
	result := InitializeVector(v.Size())
	for i := 0; i < v.Size(); i++ {
		result.Values[i] = applier(v.Values[i])
	}
	return result
}
