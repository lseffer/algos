package matrix

import (
	"errors"
	"fmt"
	"math"
)

type matrix interface {
	Add(other matrix) matrix
	AddConstant(constant float64) matrix
	MultiplyConstant(constant float64) matrix
	Multiply(other matrix) matrix
	Dims() (int, int)
	Transpose() matrix
}

// DenseMatrix is a normal dense matrix
type DenseMatrix struct {
	Rows []*Vector
}

// InitializeMatrix an empty matrix of the specified size
func InitializeMatrix(rows, cols int) *DenseMatrix {
	if rows < 0 {
		rows = 0
	}
	if cols < 0 {
		cols = 0
	}
	var vec *Vector
	matrix := make([]*Vector, rows)
	for i := 0; i < rows; i++ {
		vec = InitializeVector(cols)
		matrix[i] = vec
	}
	return &DenseMatrix{Rows: matrix}
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
	if len(m.Rows) == 0 {
		return 0, 0
	}
	thisRows := len(m.Rows)
	thisCols := m.Rows[0].Size()
	return thisRows, thisCols
}

// Tranpose transposes the matrix and returns a new one
func (m *DenseMatrix) Tranpose() *DenseMatrix {
	rows, cols := m.Dims()
	result := InitializeMatrix(cols, rows)
	for rowIndex, row := range m.Rows {
		for colIndex, element := range row.Values {
			result.Rows[colIndex].Values[rowIndex] = element
		}
	}
	return result
}

// AddConstant add constant to all elements of matrix
func (m *DenseMatrix) AddConstant(constant float64) *DenseMatrix {
	rows, cols := m.Dims()
	result := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec = rowVector.AddConstant(constant)
		result.Rows[i] = vec
	}
	return result
}

// MultiplyConstant multiply constant to all elements of matrix
func (m *DenseMatrix) MultiplyConstant(constant float64) *DenseMatrix {
	rows, cols := m.Dims()
	result := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec = rowVector.MultiplyConstant(constant)
		result.Rows[i] = vec
	}
	return result
}

type applier func(float64) float64

// ApplyFunc apply function to all elements of matrix
func (m *DenseMatrix) ApplyFunc(applier applier) *DenseMatrix {
	rows, cols := m.Dims()
	result := InitializeMatrix(rows, cols)
	var vec *Vector
	for i, rowVector := range m.Rows {
		vec = rowVector.ApplyFunc(applier)
		result.Rows[i] = vec
	}
	return result
}

// ReduceSum sum all elements in an axis and return the resulting vector. Axis=0 along columns, axis=1 along rows.
func (m *DenseMatrix) ReduceSum(axis int) (result *Vector) {
	rows, cols := m.Dims()
	if axis == 0 {
		result = InitializeVector(rows)
	} else {
		result = InitializeVector(cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if axis == 0 {
				result.Values[i] += m.Rows[i].Values[j]
			} else {
				result.Values[j] += m.Rows[i].Values[j]
			}
		}
	}
	return result
}

// Add two matrices together
func (m *DenseMatrix) Add(other *DenseMatrix) (result *DenseMatrix, err error) {
	thisRows, thisCols := m.Dims()
	otherRows, otherCols := other.Dims()
	result = InitializeMatrix(thisRows, thisCols)
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
func (m *DenseMatrix) Multiply(other *DenseMatrix) (result *DenseMatrix, err error) {
	thisRows, thisCols := m.Dims()
	otherRows, otherCols := other.Dims()
	result = InitializeMatrix(thisRows, otherCols)
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

// GetSubset get a subset of the matrix. Axis=0 along rows, axis=1 along columns.
func (m *DenseMatrix) GetSubset(startIndex, endIndex, axis int) (*DenseMatrix, error) {
	if axis != 0 && axis != 1 {
		return &DenseMatrix{Rows: nil}, errors.New("Axis can only be 0 or 1")
	}
	thisRows, thisCols := m.Dims()
	if axis == 0 && (startIndex < 0 || startIndex > thisRows || endIndex < startIndex) {
		return &DenseMatrix{Rows: nil}, fmt.Errorf("Row indices (%v, %v) out of bounds", startIndex, endIndex)
	}
	if axis == 1 && (startIndex < 0 || startIndex > thisCols || endIndex < startIndex) {
		return &DenseMatrix{Rows: nil}, fmt.Errorf("Column indices (%v, %v) out of bounds", startIndex, endIndex)
	}
	numberElements := endIndex - startIndex + 1
	var result *DenseMatrix
	var err error
	if axis == 0 {
		result = InitializeMatrix(numberElements, thisCols)
		for i := 0; i < numberElements; i++ {
			result.Rows[0] = m.Rows[startIndex+i]
		}
	} else {
		result = InitializeMatrix(thisRows, numberElements)
		for i, rowVector := range m.Rows {
			result.Rows[i].Values = rowVector.Values[startIndex : endIndex+1]
		}
	}
	return result, err
}

// Qualifier is a type of function for qualifying a matrix row for a binary split
type Qualifier func(rowIndex int, values *Vector) bool

// Split a matrix into two by applying a function to each row. Returns the row indices of the split
func Split(X *DenseMatrix, qualifier Qualifier) (left []int, right []int) {
	var qualified bool
	for rowIndex, rowVector := range X.Rows {
		qualified = qualifier(rowIndex, rowVector)
		if qualified {
			left = append(left, rowIndex)
		} else {
			right = append(right, rowIndex)
		}
	}
	return left, right
}

// GetSubSetByIndex create a new matrix from a subset of indices
func GetSubSetByIndex(X *DenseMatrix, indices []int) (res *DenseMatrix) {
	_, cols := X.Dims()
	res = InitializeMatrix(len(indices), cols)
	for newIndex, originalIndex := range indices {
		res.Rows[newIndex] = X.Rows[originalIndex]
	}
	return
}

// GetSubSetByColIndex create a new matrix from a subset of indices
func GetSubSetByColIndex(X *DenseMatrix, indices []int) (res *DenseMatrix) {
	Xnew := X.Tranpose()
	tempRes := GetSubSetByIndex(Xnew, indices)
	res = tempRes.Tranpose()
	return
}

// MinMax returns the maximum and minimum value for each column
func (m *DenseMatrix) MinMax() (minArray, maxArray []float64) {
	cols, _ := m.Dims()
	minArray = make([]float64, cols)
	maxArray = make([]float64, cols)
	for rowIndex, rowVector := range m.Rows {
		for colIndex, colValue := range rowVector.Values {
			if rowIndex == 0 {
				maxArray[colIndex] = math.Inf(-1)
				minArray[colIndex] = math.Inf(1)
			}
			if maxArray[colIndex] < colValue {
				maxArray[colIndex] = colValue
			}
			if minArray[colIndex] > colValue {
				minArray[colIndex] = colValue
			}
		}
	}
	return
}
