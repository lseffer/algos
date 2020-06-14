package ml

import (
	"algos/matrix"
	"errors"
	"sort"
)

// ClassValue a single value of a class encoding, typically one of 0, 1, 3 ...
type ClassValue float64

// ClassVector holds information about matrix containing class values
type ClassVector struct {
	Values        *matrix.DenseMatrix
	Counter       ClassCounter
	MajorityClass ClassValue
}

// NewClassVector creates a new ClassVector struct from a matrix.
// The ClassVector will be constructed from the last column in the matrix.
func NewClassVector(X *matrix.DenseMatrix) (ClassVector, error) {
	_, cols := X.Dims()
	var classValues *matrix.DenseMatrix
	var classCounter ClassCounter
	var result ClassVector
	var err error
	classValues, err = X.GetSubset(cols-1, cols-1, 1)
	classCounter, err = newClassCounter(classValues)
	result.MajorityClass, err = getMajorityClass(classCounter)
	result.Values = classValues
	result.Counter = classCounter
	return result, err
}

// ClassCounter is a map that counts the occurences of each class
type ClassCounter map[ClassValue]int

func newClassCounter(X *matrix.DenseMatrix) (ClassCounter, error) {
	var classVal ClassValue
	result := make(map[ClassValue]int)
	_, cols := X.Dims()
	if cols != 1 {
		return result, errors.New("Input matrix must be a 1-D column matrix")
	}
	for _, rowVector := range X.Rows {
		classVal = ClassValue(rowVector.Values[0])
		if _, ok := result[classVal]; ok {
			result[classVal]++
		} else {
			result[classVal] = 1
		}
	}
	return result, nil
}

// getMajorityClass gets the class with the largest count from the classcounter map.
// Keys will be sorted in ascending order before finding the majority class.
func getMajorityClass(classCounter ClassCounter) (ClassValue, error) {
	if len(classCounter) <= 0 {
		return -1.0, errors.New("Empty classCounter map")
	}
	keys := make([]float64, 0)
	for k := range classCounter {
		keys = append(keys, float64(k))
	}
	sort.Float64s(keys)
	var maxKey float64
	var val int
	maxCount := 0
	for _, key := range keys {
		val = classCounter[ClassValue(key)]
		if val > maxCount {
			maxKey = key
			maxCount = val
		}
	}
	return ClassValue(maxKey), nil
}
