package matrix

import (
	"errors"
	"fmt"
)

type matrix interface {
	Add(other matrix) matrix
	AddConstant(constant float32) matrix
	Multiply(other matrix) matrix
	Dims() (int, int)
	Transpose() matrix
}

type applier func(float32) float32

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
	rows, cols := m.Dims()
	result := ""
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result += fmt.Sprintf("| %v ", m.Rows[i][j])
		}
		result += "|\n"
	}
	return result
}

// Dims get the dimensions of the matrix
func (m *DenseMatrix) Dims() (int, int) {
	thisRows := len(m.Rows)
	thisCols := len(m.Rows[0])
	return thisRows, thisCols
}

// Tranpose transposes the matrix and returns a new one
func (m *DenseMatrix) Tranpose() (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(cols, rows)
	for rowIndex, row := range m.Rows {
		for colIndex, element := range row {
			result.Rows[colIndex][rowIndex] = element
		}
	}
	return result, err
}

// AddConstant add constant to all elements of matrix
func (m *DenseMatrix) AddConstant(constant float32) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Rows[i][j] = m.Rows[i][j] + constant
		}
	}
	return result, err
}

// MultiplyConstant multiply constant to all elements of matrix
func (m *DenseMatrix) MultiplyConstant(constant float32) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Rows[i][j] = m.Rows[i][j] * constant
		}
	}
	return result, err
}

// ApplyFunc apply function to all elements of matrix
func (m *DenseMatrix) ApplyFunc(applier applier) (*DenseMatrix, error) {
	rows, cols := m.Dims()
	result, err := InitializeMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Rows[i][j] = applier(m.Rows[i][j])
		}
	}
	return result, err
}

// ReduceSum sum all elements in an axis and return the resulting vector
func (m *DenseMatrix) ReduceSum(axis int) (*DenseMatrix, error) {
	if axis > 1 || axis < 0 {
		return &DenseMatrix{}, errors.New("Axis out of bounds, must be 0 or 1")
	}
	rows, cols := m.Dims()
	var resultRows int
	var resultCols int
	if axis == 0 {
		resultRows = rows
		resultCols = 1
	} else {
		resultRows = 1
		resultCols = cols
	}
	result, err := InitializeMatrix(resultRows, resultCols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if axis == 0 {
				result.Rows[i][0] += m.Rows[i][j]
			} else {
				result.Rows[0][j] += m.Rows[i][j]
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
			result.Rows[i][j] = m.Rows[i][j] + other.Rows[i][j]
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
				result.Rows[i][j] += m.Rows[i][k] * other.Rows[k][j]
			}
		}
	}
	return result, err
}
