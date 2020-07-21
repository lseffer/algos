package trees

import (
	"testing"

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
