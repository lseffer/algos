package trees

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/stretchr/testify/assert"
)

func TestExtraTreesClassifierFitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewExtraTreesClassifier(1, 1, GiniCriteria{}, 10)
	model.Fit(testdata)
	for _, tree := range model.trees {
		assert.NotNil(t, tree.rootNode)
	}
	assert.Equal(t, 10, len(model.trees))
	model.Predict(testdata.Features)
}

func TestExtraTreesRegressorFitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewExtraTreesRegressor(1, 1, 10)
	model.Fit(testdata)
	for _, tree := range model.trees {
		assert.NotNil(t, tree.rootNode)
	}
	assert.Equal(t, 10, len(model.trees))
	model.Predict(testdata.Features)
}
