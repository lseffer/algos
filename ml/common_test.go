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
