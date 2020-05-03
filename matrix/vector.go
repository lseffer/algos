package matrix

import (
	"fmt"
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
