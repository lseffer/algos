package trees

// import (
// 	"testing"

// 	"github.com/lseffer/algos/pkg/ml"
// )

// func Test_entropyCriteria_formula(t *testing.T) {
// 	type args struct {
// 		classCounter ml.ClassCounter
// 		totalCount   int
// 	}
// 	tests := []struct {
// 		name    string
// 		c       EntropyCriteria
// 		args    args
// 		want    float64
// 		wantErr bool
// 	}{
// 		{
// 			name: "only 1 class",
// 			args: args{
// 				totalCount:   10,
// 				classCounter: ml.ClassCounter{0.0: 10, 1.0: 0, 2.0: 0},
// 			},
// 			want:    0.0,
// 			wantErr: false,
// 		},
// 		{
// 			name: "even spread",
// 			args: args{
// 				totalCount:   4,
// 				classCounter: ml.ClassCounter{0.0: 2, 1.0: 2},
// 			},
// 			want:    1.0,
// 			wantErr: false,
// 		},
// 		{
// 			name: "faulty input",
// 			args: args{
// 				totalCount:   0,
// 				classCounter: ml.ClassCounter{0.0: 2, 1.0: 2},
// 			},
// 			want:    0.0,
// 			wantErr: false,
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			c := EntropyCriteria{}
// 			got, err := c.formula(tt.args.classCounter, tt.args.totalCount)
// 			if (err != nil) != tt.wantErr {
// 				t.Errorf("entropyCriteria.formula() error = %v, wantErr %v", err, tt.wantErr)
// 				return
// 			}
// 			if got != tt.want {
// 				t.Errorf("entropyCriteria.formula() = %v, want %v", got, tt.want)
// 			}
// 		})
// 	}
// }

// func Test_giniCriteria_formula(t *testing.T) {
// 	type args struct {
// 		classCounter ml.ClassCounter
// 		totalCount   int
// 	}
// 	tests := []struct {
// 		name    string
// 		c       GiniCriteria
// 		args    args
// 		want    float64
// 		wantErr bool
// 	}{
// 		{
// 			name: "only 1 class",
// 			args: args{
// 				totalCount:   10,
// 				classCounter: ml.ClassCounter{0.0: 10, 1.0: 0, 2.0: 0},
// 			},
// 			want:    0.0,
// 			wantErr: false,
// 		},
// 		{
// 			name: "even spread",
// 			args: args{
// 				totalCount:   4,
// 				classCounter: ml.ClassCounter{0.0: 2, 1.0: 2},
// 			},
// 			want:    0.5,
// 			wantErr: false,
// 		},
// 		{
// 			name: "faulty input",
// 			args: args{
// 				totalCount:   0,
// 				classCounter: ml.ClassCounter{0.0: 2, 1.0: 2},
// 			},
// 			want:    0.0,
// 			wantErr: false,
// 		},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			c := GiniCriteria{}
// 			got, err := c.formula(tt.args.classCounter, tt.args.totalCount)
// 			if (err != nil) != tt.wantErr {
// 				t.Errorf("giniCriteria.formula() error = %v, wantErr %v", err, tt.wantErr)
// 				return
// 			}
// 			if got != tt.want {
// 				t.Errorf("giniCriteria.formula() = %v, want %v", got, tt.want)
// 			}
// 		})
// 	}
// }
