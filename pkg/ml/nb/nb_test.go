package nb

import (
	"math"
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/lseffer/algos/pkg/ml"
	"github.com/stretchr/testify/assert"
)

func TestNBClassifier(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewNaiveBayes()
	model.Fit(testdata)
	res := model.Predict(testdata.Features)
	assert.Equal(t, 0.0, res.Rows[0].Values[0])
	assert.Equal(t, 0.0, res.Rows[1].Values[0])
	assert.Equal(t, 0.0, res.Rows[2].Values[0])
	assert.Equal(t, []float64{2.0, 2.5, 3.5}, model.means[ml.TargetValue(0.0)])
	assert.Equal(t, []float64{4.0, 5.0, 8.0}, model.means[ml.TargetValue(1.0)])
	assert.Equal(t, []float64{2.0, 4.5, 12.5}, model.vars[ml.TargetValue(0.0)])
	assert.True(t, math.IsNaN(model.vars[ml.TargetValue(1.0)][0]))
}

func TestPredictClass(t *testing.T) {
	model := NewNaiveBayes()
	assert.Equal(t, ml.TargetValue(1.0), model.predictClass(map[ml.TargetValue]float64{ml.TargetValue(1.0): 0.3, ml.TargetValue(2.0): 0.05, ml.TargetValue(3.0): 0.1}))
}

func TestGetMeanAndVariance(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewNaiveBayes()
	mean, variance := model.getMeanAndVariance(testdata.Features)
	assert.InDeltaSlice(t, []float64{2.6, 3.33, 5.0}, mean, 0.1)
	assert.InDeltaSlice(t, []float64{2.33, 4.33, 13.0}, variance, 0.1)
}
