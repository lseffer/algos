package trees

import (
	"testing"

	"github.com/lseffer/algos/internal/pkg/mltest"
	"github.com/stretchr/testify/assert"
)

func TestRFCitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewRandomForestClassifier(1, 1, GiniCriteria{}, GreedySplitFinder{}, 10)
	model.Fit(testdata)
	for _, tree := range model.trees {
		assert.NotNil(t, tree.rootNode)
	}
	assert.Equal(t, 10, len(model.trees))
	model.Predict(testdata.Features)
}

func TestRFRitPredict(t *testing.T) {
	testdata := mltest.DataSample()
	model := NewRandomForestRegressor(1, 1, GreedySplitFinder{}, 10)
	model.Fit(testdata)
	for _, tree := range model.trees {
		assert.NotNil(t, tree.rootNode)
	}
	assert.Equal(t, 10, len(model.trees))
	model.Predict(testdata.Features)
}
