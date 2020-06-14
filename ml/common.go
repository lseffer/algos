package ml

import (
	"algos/matrix"
	"errors"
)

// ClassValue a single value of a class encoding, typically one of 0, 1, 3 ...
type ClassValue float64

type classes []ClassValue

// ClassVector holds information about matrix containing class values
type ClassVector struct {
	Values        *matrix.DenseMatrix
	Classes       classes
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
	result.Classes = getClasses(classCounter)
	result.Values = X
	result.Counter = classCounter
	return result, err
}

// ClassCounter is a map that counts the occurences of each class
type ClassCounter map[ClassValue]int

func getClasses(classCounter ClassCounter) classes {
	var result classes
	result = make([]ClassValue, 1)
	for key := range classCounter {
		result = append(result, key)
	}
	return result
}

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

func getMajorityClass(classCounter ClassCounter) (ClassValue, error) {
	if len(classCounter) <= 0 {
		return -1.0, errors.New("Empty classCounter map")
	}
	var maxKey ClassValue
	maxCount := 0
	for key, val := range classCounter {
		if val > maxCount {
			maxKey = key
		}
	}
	return maxKey, nil
}
