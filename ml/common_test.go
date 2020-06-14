package ml

import (
	"algos/matrix"
	"reflect"
	"testing"
)

func Test_newClassCounter(t *testing.T) {
	var mRow1 = matrix.Vector{Values: []float64{0}}
	var mRow2 = matrix.Vector{Values: []float64{0}}
	var mRow3 = matrix.Vector{Values: []float64{1}}
	var mRow4 = matrix.Vector{Values: []float64{0}}
	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
	type args struct {
		X *matrix.DenseMatrix
	}
	tests := []struct {
		name    string
		args    args
		want    ClassCounter
		wantErr bool
	}{
		{
			name: "normal case",
			args: args{
				X: &testMat,
			},
			want:    ClassCounter{0.0: 3, 1.0: 1},
			wantErr: false,
		},
		{
			name: "empty matrix",
			args: args{
				X: &matrix.DenseMatrix{Rows: []*matrix.Vector{}},
			},
			want:    ClassCounter{},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := newClassCounter(tt.args.X)
			if (err != nil) != tt.wantErr {
				t.Errorf("newClassCounter() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newClassCounter() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getMajorityClass(t *testing.T) {
	type args struct {
		classCounter ClassCounter
	}
	tests := []struct {
		name    string
		args    args
		want    ClassValue
		wantErr bool
	}{
		{
			name: "normal case",
			args: args{
				classCounter: ClassCounter{0.0: 1, 2.0: 3},
			},
			want:    ClassValue(2.0),
			wantErr: false,
		},
		{
			name: "normal case 2",
			args: args{
				classCounter: ClassCounter{0.0: 100, 2.0: 3},
			},
			want:    ClassValue(0.0),
			wantErr: false,
		},
		{
			name: "counts equal, should return first",
			args: args{
				classCounter: ClassCounter{0.0: 3, 2.0: 3},
			},
			want:    ClassValue(0.0),
			wantErr: false,
		},
		{
			name: "empty counter",
			args: args{
				classCounter: ClassCounter{},
			},
			want:    -1.0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getMajorityClass(tt.args.classCounter)
			if (err != nil) != tt.wantErr {
				t.Errorf("getMajorityClass() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("getMajorityClass() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewClassVector(t *testing.T) {
	var mRow1 = matrix.Vector{Values: []float64{0, 1}}
	var mRow2 = matrix.Vector{Values: []float64{0, 2}}
	var mRow3 = matrix.Vector{Values: []float64{1, 2}}
	var mRow4 = matrix.Vector{Values: []float64{0, 1}}
	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
	result, _ := testMat.GetSubset(1, 1, 1)
	type args struct {
		X *matrix.DenseMatrix
	}
	tests := []struct {
		name    string
		args    args
		want    ClassVector
		wantErr bool
	}{
		{
			name: "Normal case",
			args: args{
				X: &testMat,
			},
			want: ClassVector{Values: result, Counter: ClassCounter{1.0: 2, 2.0: 2}, MajorityClass: ClassValue(1.0)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewClassVector(tt.args.X)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClassVector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewClassVector() = %v, want %v", got, tt.want)
			}
		})
	}
}
