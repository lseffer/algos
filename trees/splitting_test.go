package trees

import (
	"algos/matrix"
	"algos/ml"
	"testing"
)

func Test_scoreSplit(t *testing.T) {
	var mRow1 = matrix.Vector{Values: []float64{0}}
	var mRow2 = matrix.Vector{Values: []float64{0}}
	var mRow3 = matrix.Vector{Values: []float64{1}}
	var mRow4 = matrix.Vector{Values: []float64{0}}
	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2, &mRow3, &mRow4}}
	var testMat1 = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2}}
	var testMatEmpty = matrix.DenseMatrix{Rows: []*matrix.Vector{}}
	testCVNormal, _ := ml.NewClassVector(&testMat)
	testCVEmpty, _ := ml.NewClassVector(&testMatEmpty)
	testCVPerfectSplit, _ := ml.NewClassVector(&testMat1)
	type args struct {
		left     ml.ClassVector
		right    ml.ClassVector
		criteria splitCriteria
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			name: "normal case",
			args: args{
				left:     testCVNormal,
				right:    testCVNormal,
				criteria: giniCriteria{},
			},
			want:    0.375,
			wantErr: false,
		},
		{
			name: "right split zero length",
			args: args{
				left:     testCVNormal,
				right:    testCVEmpty,
				criteria: giniCriteria{},
			},
			want:    0.375,
			wantErr: false,
		},
		{
			name: "left split zero length",
			args: args{
				left:     testCVEmpty,
				right:    testCVNormal,
				criteria: giniCriteria{},
			},
			want:    0.375,
			wantErr: false,
		},
		{
			name: "perfect split",
			args: args{
				left:     testCVEmpty,
				right:    testCVPerfectSplit,
				criteria: giniCriteria{},
			},
			want:    0.0,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scoreSplit(tt.args.left, tt.args.right, tt.args.criteria)
			if (err != nil) != tt.wantErr {
				t.Errorf("scoreSplit() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("scoreSplit() = %v, want %v", got, tt.want)
			}
		})
	}
}
