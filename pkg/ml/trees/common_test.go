package trees

import (
	"reflect"
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/ml"
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

func TestClassificationPredictor_predict(t *testing.T) {
	type args struct {
		data ml.DataSet
	}
	tests := []struct {
		name string
		c    ClassificationPredictor
		args args
		want ml.TargetValue
	}{
		{
			name: "Should equal majority class",
			c:    ClassificationPredictor{},
			args: args{
				data: mltest.DataSample(),
			},
			want: ml.TargetValue(0.0),
		},
		{
			name: "Equal number of classes; should equal the numerically smaller one",
			c:    ClassificationPredictor{},
			args: args{
				data: mltest.DataSampleEvenSpread(),
			},
			want: ml.TargetValue(0.0),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := ClassificationPredictor{}
			if got := c.predict(tt.args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ClassificationPredictor.predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRegressionPredicor_predict(t *testing.T) {
	type args struct {
		data ml.DataSet
	}
	tests := []struct {
		name           string
		c              RegressionPredicor
		args           args
		wantPrediction ml.TargetValue
	}{
		{
			name: "Should equal majority class",
			c:    RegressionPredicor{},
			args: args{
				data: mltest.DataSample(),
			},
			wantPrediction: ml.TargetValue(1.0 / 3.0),
		},
		{
			name: "Only one class",
			c:    RegressionPredicor{},
			args: args{
				data: mltest.DataSampleOneClass(),
			},
			wantPrediction: ml.TargetValue(0.0),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := RegressionPredicor{}
			if gotPrediction := c.predict(tt.args.data); !reflect.DeepEqual(gotPrediction, tt.wantPrediction) {
				t.Errorf("RegressionPredicor.predict() = %v, want %v", gotPrediction, tt.wantPrediction)
			}
		})
	}
}
