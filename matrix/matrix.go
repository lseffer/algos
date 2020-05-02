package matrix

import (
	"errors"
	"fmt"
)

type matrix interface {
	Add(other matrix) matrix
	AddConstant(constant float32) matrix
	Multiply(other matrix) matrix
	Dims() [2]int
	Transpose() matrix
}

// DenseMatrix is a normal matrix
type DenseMatrix struct {
	Rows [][]float32
}

// InitializeMatrix an empty matrix of the specified size
func InitializeMatrix(rows, cols int) (*DenseMatrix, error) {
	if rows < 1 || cols < 1 {
		return &DenseMatrix{}, fmt.Errorf("Rows and columns must be greater than 0")
	}
	matrix := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float32, cols)
	}
	return &DenseMatrix{Rows: matrix}, nil
}

// Dims get the dimensions of the matrix
func (m DenseMatrix) String() string {
	dims := m.Dims()
	result := ""
	for i := 0; i < dims[0]; i++ {
		for j := 0; j < dims[1]; j++ {
			result += fmt.Sprintf("| %v ", m.Rows[i][j])
		}
		result += "|\n"
	}
	return result
}

// Dims get the dimensions of the matrix
func (m *DenseMatrix) Dims() [2]int {
	thisRows := len(m.Rows)
	thisCols := len(m.Rows[0])
	return [2]int{thisRows, thisCols}
}

// Tranpose transposes the matrix and returns a new one
func (m *DenseMatrix) Tranpose() (*DenseMatrix, error) {
	dims := m.Dims()
	result, err := InitializeMatrix(dims[1], dims[0])
	for rowIndex, row := range m.Rows {
		for colIndex, element := range row {
			result.Rows[colIndex][rowIndex] = element
		}
	}
	return result, err
}

// AddConstant add constant to all elements of matrix
func (m *DenseMatrix) AddConstant(constant float32) (*DenseMatrix, error) {
	thisDims := m.Dims()
	rows := thisDims[0]
	cols := thisDims[1]
	result, err := InitializeMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Rows[i][j] = m.Rows[i][j] + constant
		}
	}
	return result, err
}

// Add two matrices together
func (m *DenseMatrix) Add(other *DenseMatrix) (*DenseMatrix, error) {
	thisDims := m.Dims()
	otherdims := other.Dims()
	rows := thisDims[0]
	cols := thisDims[1]
	result, err := InitializeMatrix(rows, cols)
	if thisDims != otherdims {
		return result, errors.New("dimensions to do not match")
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Rows[i][j] = m.Rows[i][j] + other.Rows[i][j]
		}
	}
	return result, err
}

// Multiply the matrix with another matrix
func (m *DenseMatrix) Multiply(other *DenseMatrix) (*DenseMatrix, error) {
	thisDims := m.Dims()
	otherDims := other.Dims()
	thisRows := thisDims[0]
	thisCols := thisDims[1]
	otherRows := otherDims[0]
	otherCols := otherDims[1]
	result, err := InitializeMatrix(thisRows, otherCols)
	if thisCols != otherRows {
		return result, fmt.Errorf("Dimensions of matrices are incompatible for multiplication: %v,  %v", thisDims, otherDims)
	}
	for i := 0; i < thisRows; i++ {
		for j := 0; j < otherCols; j++ {
			for k := 0; k < thisCols; k++ {
				result.Rows[i][j] += m.Rows[i][k] * other.Rows[k][j]
			}
		}
	}
	return result, err
}
