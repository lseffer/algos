package trees

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/ml"
)

func Test_scoreSplit(t *testing.T) {
	type args struct {
		left     ml.DataSet
		right    ml.DataSet
		criteria splitCriteria
	}
	tests := []struct {
		name      string
		args      args
		wantScore float64
	}{
		{
			name: "Test with same dataset, perfect score",
			args: args{left: mltest.DataSampleOneClass(),
				right:    mltest.DataSampleOneClass(),
				criteria: GiniCriteria{},
			},
			wantScore: 0.0,
		},
		{
			name: "Test with same dataset, gini gets 0.5",
			args: args{left: mltest.DataSampleEvenSpread(),
				right:    mltest.DataSampleEvenSpread(),
				criteria: GiniCriteria{},
			},
			wantScore: 0.5,
		},
		{
			name: "Even spread, max score for entropy",
			args: args{left: mltest.DataSampleEvenSpread(),
				right:    mltest.DataSampleEvenSpread(),
				criteria: EntropyCriteria{},
			},
			wantScore: 1.0,
		},
		{
			name: "One class only, minimum score for entropy",
			args: args{left: mltest.DataSampleOneClass(),
				right:    mltest.DataSampleOneClass(),
				criteria: EntropyCriteria{},
			},
			wantScore: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotScore := scoreSplit(tt.args.left, tt.args.right, tt.args.criteria); gotScore != tt.wantScore {
				t.Errorf("scoreSplit() = %v, want %v", gotScore, tt.wantScore)
			}
		})
	}
}

func TestSplitFinder_algorithm(t *testing.T) {
	type args struct {
		data      ml.DataSet
		criteria  splitCriteria
		rowsStart int
		rowsEnd   int
	}
	type res struct {
		colIndex   int
		splitValue float64
	}
	tests := []struct {
		name    string
		f       splitFinder
		args    args
		wantRes res
	}{
		{
			name: "Test that we get the correct split, should be first column and the value that exceeds the split",
			f:    GreedySplitFinder{},
			args: args{
				data:      mltest.DataSampleEvenSpread(),
				criteria:  GiniCriteria{},
				rowsStart: 0,
				rowsEnd:   4,
			},
			wantRes: res{
				colIndex:   0,
				splitValue: 4.0,
			},
		},
		{
			name: "Same test as above, but now we try it with the concurrent splitfinder wrapping the greedy one",
			f:    ConcurrentSplitFinder{Jobs: 2, SplitFinder: GreedySplitFinder{}},
			args: args{
				data:      mltest.DataSampleEvenSpread(),
				criteria:  GiniCriteria{},
				rowsStart: 0,
				rowsEnd:   4,
			},
			wantRes: res{
				colIndex:   0,
				splitValue: 4.0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := GreedySplitFinder{}
			gotRes := f.algorithm(tt.args.data, tt.args.criteria, tt.args.rowsStart, tt.args.rowsEnd)
			assert.Equal(t, 0, gotRes.colIndex)
			assert.Equal(t, 4.0, gotRes.splitValue)
			if gotScore := f.algorithm(tt.args.data, tt.args.criteria, tt.args.rowsStart, tt.args.rowsEnd); gotScore.colIndex != tt.wantRes.colIndex || gotScore.splitValue != tt.wantRes.splitValue {
				t.Errorf("scoreSplit() = %v, want %v", gotScore, tt.wantRes)
			}
		})
	}
}
