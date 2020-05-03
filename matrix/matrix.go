package matrix

import (
	"errors"
	"fmt"
)

type matrix interface {
	Add(other matrix) matrix
	AddConstant(constant float32) matrix
	MultiplyConstant(constant float32) matrix
	Multiply(other matrix) matrix
	Dims() (int, int)
	Transpose() matrix
}

type applier func(float32) float32

// DenseMatrix is a normal matrix
type DenseMatrix struct {
	Rows []*Vector
}

// InitializeMatrix an empty matrix of the specified size
func InitializeMatrix(rows, cols int) (*DenseMatrix, error) {
	if rows < 1 || cols < 1 {
		return &DenseMatrix{}, fmt.Errorf("Rows and columns must be greater than 0")
	}
	var err error
	var vec *Vector
	matrix := make([]*Vector, rows)
	for i := 0; i < rows; i++ {
		vec, err = InitializeVector(cols)
		matrix[i] = vec
	}
	return &DenseMatrix{Rows: matrix}, err
}

// String representation of the matrix
func (m DenseMatrix) String() string {
	result := ""
	for _, rowVector := range m.Rows {
		result += rowVector.String()
		result += "\n"
	}
	return result
}

// Dims get the dimensions of the matrix
func (m *DenseMatrix) Dims() (int, int) {
	thisRows := len(m.Rows)
	thisCols := m.Rows[0].Size()
	return thisRows, thisCols
}

// Tranpose transposes the matrix and returns a new one
func (m *DenseMatrix) Tranpose() (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(cols, rows)
	for rowIndex, row := range m.Rows {
		for colIndex, element := range row.Values {
			result.Rows[colIndex].Values[rowIndex] = element
		}
	}
	return result, err
}

// AddConstant add constant to all elements of matrix
func (m *DenseMatrix) AddConstant(constant float32) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec, err = rowVector.AddConstant(constant)
		result.Rows[i] = vec
	}
	return result, err
}

// MultiplyConstant multiply constant to all elements of matrix
func (m *DenseMatrix) MultiplyConstant(constant float32) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec, err = rowVector.MultiplyConstant(constant)
		result.Rows[i] = vec
	}
	return result, err
}

// ApplyFunc apply function to all elements of matrix
func (m *DenseMatrix) ApplyFunc(applier applier) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec, err = rowVector.ApplyFunc(applier)
		result.Rows[i] = vec
	}
	return result, err
}

// ReduceSum sum all elements in an axis and return the resulting vector
func (m *DenseMatrix) ReduceSum(axis int) (*DenseMatrix, error) {
	if axis > 1 || axis < 0 {
		return &DenseMatrix{}, errors.New("Axis out of bounds, must be 0 or 1")
	}
	rows, cols := m.Dims()
	var result *DenseMatrix
	var err error
	if axis == 0 {
		result, err = InitializeMatrix(rows, 1)
	} else {
		result, err = InitializeMatrix(1, cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if axis == 0 {
				result.Rows[i].Values[0] += m.Rows[i].Values[j]
			} else {
				result.Rows[0].Values[j] += m.Rows[i].Values[j]
			}
		}
	}
	return result, err
}

// Add two matrices together
func (m *DenseMatrix) Add(other *DenseMatrix) (*DenseMatrix, error) {
	thisRows, thisCols := m.Dims()
	otherRows, otherCols := other.Dims()
	result, err := InitializeMatrix(thisRows, thisCols)
	if !(thisRows == otherRows && thisCols == otherCols) {
		return result, errors.New("dimensions to do not match")
	}
	for i := 0; i < thisRows; i++ {
		for j := 0; j < thisCols; j++ {
			result.Rows[i].Values[j] = m.Rows[i].Values[j] + other.Rows[i].Values[j]
		}
	}
	return result, err
}

// Multiply the matrix with another matrix
func (m *DenseMatrix) Multiply(other *DenseMatrix) (*DenseMatrix, error) {
	thisRows, thisCols := m.Dims()
	otherRows, otherCols := other.Dims()
	result, err := InitializeMatrix(thisRows, otherCols)
	if thisCols != otherRows {
		return result, fmt.Errorf(`Dimensions of matrices are incompatible
			for multiplication: (%v, %v),  (%v, %v)`, thisRows, thisCols, otherRows, otherCols)
	}
	for i := 0; i < thisRows; i++ {
		for j := 0; j < otherCols; j++ {
			for k := 0; k < thisCols; k++ {
				result.Rows[i].Values[j] += m.Rows[i].Values[k] * other.Rows[k].Values[j]
			}
		}
	}
	return result, err
}
