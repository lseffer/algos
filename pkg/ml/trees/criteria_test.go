package trees

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

func TestGiniCriteria_formula(t *testing.T) {
	type args struct {
		data ml.DataSet
	}
	tests := []struct {
		name string
		c    GiniCriteria
		args args
		want float64
	}{
		{
			name: "only 1 class",
			args: args{
				data: mltest.DataSampleOneClass(),
			},
			want: 0.0,
		},
		{
			name: "even spread",
			args: args{
				data: mltest.DataSampleEvenSpread(),
			},
			want: 0.5,
		},
		{
			name: "empty input",
			args: args{
				data: ml.DataSet{Features: matrix.InitializeMatrix(0, 0), Target: matrix.InitializeMatrix(0, 0)},
			},
			want: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := GiniCriteria{}
			if got := c.formula(tt.args.data); got != tt.want {
				t.Errorf("GiniCriteria.formula() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEntropyCriteria_formula(t *testing.T) {
	type args struct {
		data ml.DataSet
	}
	tests := []struct {
		name string
		c    EntropyCriteria
		args args
		want float64
	}{
		{
			name: "only 1 class",
			args: args{
				data: mltest.DataSampleOneClass(),
			},
			want: 0.0,
		},
		{
			name: "even spread",
			args: args{
				data: mltest.DataSampleEvenSpread(),
			},
			want: 1.0,
		},
		{
			name: "empty input",
			args: args{
				data: ml.DataSet{Features: matrix.InitializeMatrix(0, 0), Target: matrix.InitializeMatrix(0, 0)},
			},
			want: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := EntropyCriteria{}
			if got := c.formula(tt.args.data); got != tt.want {
				t.Errorf("EntropyCriteria.formula() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMeanSquaredErrorCriteria_formula(t *testing.T) {
	type args struct {
		data ml.DataSet
	}
	tests := []struct {
		name    string
		c       MeanSquaredErrorCriteria
		args    args
		wantRes float64
	}{
		{
			name: "only 1 class",
			args: args{
				data: mltest.DataSampleOneClass(),
			},
			wantRes: 0.0,
		},
		{
			name: "even spread",
			args: args{
				data: mltest.DataSampleEvenSpread(),
			},
			wantRes: 0.25,
		},
		{
			name: "empty input",
			args: args{
				data: ml.DataSet{Features: matrix.InitializeMatrix(0, 0), Target: matrix.InitializeMatrix(0, 0)},
			},
			wantRes: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := MeanSquaredErrorCriteria{}
			if gotRes := c.formula(tt.args.data); gotRes != tt.wantRes {
				t.Errorf("MeanSquaredErrorCriteria.formula() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
