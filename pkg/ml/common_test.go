package ml

import (
	"reflect"
	"testing"

	"github.com/lseffer/algos/pkg/matrix"
)

func TestTrainTestSplit(t *testing.T) {
	var mRow1 = matrix.Vector{Values: []float64{1, 2}}
	var mRow2 = matrix.Vector{Values: []float64{3, 4}}
	var testMat = matrix.DenseMatrix{Rows: []*matrix.Vector{&mRow1, &mRow2}}
	empty := matrix.InitializeMatrix(0, 2)

	type args struct {
		dataset   DataSet
		trainSize float64
	}
	tests := []struct {
		name      string
		args      args
		wantTrain DataSet
		wantTest  DataSet
	}{
		{
			name:      "Test take only train on a normal dataset",
			args:      args{dataset: DataSet{Features: &testMat, Target: &testMat}, trainSize: 1.0},
			wantTest:  DataSet{Features: empty, Target: empty},
			wantTrain: DataSet{Features: &testMat, Target: &testMat},
		},
		{
			name:      "Test take only test on a normal dataset",
			args:      args{dataset: DataSet{Features: &testMat, Target: &testMat}, trainSize: 0.0},
			wantTrain: DataSet{Features: empty, Target: empty},
			wantTest:  DataSet{Features: &testMat, Target: &testMat},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTrain, gotTest := TrainTestSplit(tt.args.dataset, tt.args.trainSize)
			if !reflect.DeepEqual(gotTrain, tt.wantTrain) {
				t.Errorf("TrainTestSplit() gotTrain = %v, want %v", gotTrain, tt.wantTrain)
			}
			if !reflect.DeepEqual(gotTest, tt.wantTest) {
				t.Errorf("TrainTestSplit() gotTest = %v, want %v", gotTest, tt.wantTest)
			}
		})
	}
}
