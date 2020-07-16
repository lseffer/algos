package trees

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTreeStack(t *testing.T) {
	rootNode := &treeNode{depth: 0}
	s := make(treeStack, 0)
	newStack, node := s.Pop()
	assert.Nil(t, newStack)
	assert.Nil(t, node)
	s = s.Push(rootNode)
	assert.Equal(t, 1, len(s))
	assert.Equal(t, 1, s.Size())
}

// func Test_NewClassCounter(t *testing.T) {
// 	var mRow1 = matrix.Vector{Values: []float64{0}}
// 	var mRow2 = matrix.Vector{Values: []float64{0}}
// 	var mRow3 = matrix.Vector{Values: []float64{1}}
// 	var mRow4 = matrix.Vector{Values: []float64{0}}
// 	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
// 	type args struct {
// 		X *matrix.DenseMatrix
// 	}
// 	tests := []struct {
// 		name string
// 		args args
// 		want ClassCounter
// 	}{
// 		{
// 			name: "normal case",
// 			args: args{
// 				X: &testMat,
// 			},
// 			want: ClassCounter{0.0: 3, 1.0: 1},
// 		},
// 		{
// 			name: "empty matrix",
// 			args: args{
// 				X: &matrix.DenseMatrix{Rows: []*matrix.Vector{}},
// 			},
// 			want: ClassCounter{},
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			got := NewClassCounter(tt.args.X)
// 			if !reflect.DeepEqual(got, tt.want) {
// 				t.Errorf("NewClassCounter() = %v, want %v", got, tt.want)
// 			}
// 		})
// 	}
// }

// func Test_GetMajorityClass(t *testing.T) {
// 	type args struct {
// 		classCounter ClassCounter
// 	}
// 	tests := []struct {
// 		name string
// 		args args
// 		want ClassValue
// 	}{
// 		{
// 			name: "normal case",
// 			args: args{
// 				classCounter: ClassCounter{0.0: 1, 2.0: 3},
// 			},
// 			want: ClassValue(2.0),
// 		},
// 		{
// 			name: "normal case 2",
// 			args: args{
// 				classCounter: ClassCounter{0.0: 100, 2.0: 3},
// 			},
// 			want: ClassValue(0.0),
// 		},
// 		{
// 			name: "counts equal, should return first",
// 			args: args{
// 				classCounter: ClassCounter{0.0: 3, 2.0: 3},
// 			},
// 			want: ClassValue(0.0),
// 		},
// 		{
// 			name: "empty counter",
// 			args: args{
// 				classCounter: ClassCounter{},
// 			},
// 			want: -1.0,
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			got := GetMajorityClass(tt.args.classCounter)
// 			if got != tt.want {
// 				t.Errorf("GetMajorityClass() = %v, want %v", got, tt.want)
// 			}
// 		})
// 	}
// }

// func TestNewClassVector(t *testing.T) {
// 	var mRow1 = matrix.Vector{Values: []float64{0, 1}}
// 	var mRow2 = matrix.Vector{Values: []float64{0, 2}}
// 	var mRow3 = matrix.Vector{Values: []float64{1, 2}}
// 	var mRow4 = matrix.Vector{Values: []float64{0, 1}}
// 	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
// 	result, _ := testMat.GetSubset(1, 1, 1)
// 	type args struct {
// 		X *matrix.DenseMatrix
// 	}
// 	tests := []struct {
// 		name    string
// 		args    args
// 		want    ClassVector
// 		wantErr bool
// 	}{
// 		{
// 			name: "Normal case",
// 			args: args{
// 				X: &testMat,
// 			},
// 			want: ClassVector{Values: result, Counter: ClassCounter{1.0: 2, 2.0: 2}, MajorityClass: ClassValue(1.0)},
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			got, err := NewClassVector(tt.args.X)
// 			if (err != nil) != tt.wantErr {
// 				t.Errorf("NewClassVector() error = %v, wantErr %v", err, tt.wantErr)
// 				return
// 			}
// 			if !reflect.DeepEqual(got, tt.want) {
// 				t.Errorf("NewClassVector() = %v, want %v", got, tt.want)
// 			}
// 		})
// 	}
// }
