package matrix

import (
	"reflect"
	"testing"
)

func Test_parseMatrix(t *testing.T) {
	type args struct {
		stringArr [][]string
	}
	tests := []struct {
		name    string
		args    args
		wantRes *DenseMatrix
		wantErr bool
	}{
		{
			name: "Test loading a numeric string array",
			args: args{stringArr: [][]string{{"0", "1", "1", "2", "3"},
				{"0", "124", "14", "12", "0"},
				{"0", "34", "4", "3", "34"}}},
			wantRes: &DenseMatrix{Rows: []*Vector{{Values: []float64{0.0, 1.0, 1.0, 2.0, 3.0}}, {Values: []float64{0.0, 124.0, 14.0, 12.0, 0.0}}, {Values: []float64{0.0, 34.0, 4.0, 3.0, 34.0}}}},
			wantErr: false,
		},
		{
			name: "Test loading a string array with differing number of elements",
			args: args{stringArr: [][]string{{"0", "1", "1", "2", "3"},
				{"0", "124", "14"},
				{"0", "34", "4"}}},
			wantRes: &DenseMatrix{Rows: []*Vector{{Values: []float64{0.0, 1.0, 1.0, 2.0, 3.0}}, {Values: []float64{0.0, 124.0, 14.0, 0.0, 0.0}}, {Values: []float64{0.0, 34.0, 4.0, 0.0, 0.0}}}},
			wantErr: false,
		},
		{
			name: "Test loading a string array with non-numerics",
			args: args{stringArr: [][]string{{"hello", "1", "1", "2", "3"},
				{"0", "124", "14"},
				{"0", "34", "4"}}},
			wantRes: &DenseMatrix{Rows: []*Vector{{Values: []float64{0.0, 0.0, 0.0, 0.0, 0.0}}, {Values: []float64{0.0, 0.0, 0.0, 0.0, 0.0}}, {Values: []float64{0.0, 0.0, 0.0, 0.0, 0.0}}}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotRes, err := parseMatrix(tt.args.stringArr)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseMatrix() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotRes, tt.wantRes) {
				t.Errorf("parseMatrix() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
