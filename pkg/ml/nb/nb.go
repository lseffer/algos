package nb

import (
	"math"

	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

type NaiveBayes struct {
	Distribution string
	means        map[ml.TargetValue][]float64
	vars         map[ml.TargetValue][]float64
	priorProba   map[ml.TargetValue]float64
}

func NewNaiveBayes() NaiveBayes {
	return NaiveBayes{Distribution: "gaussian"}
}

func (m *NaiveBayes) getMeanAndVariance(X *matrix.DenseMatrix) ([]float64, []float64) {
	var diff *matrix.Vector
	rows, cols := X.Dims()
	means := X.ReduceSum(1)
	means = means.MultiplyConstant(1.0 / float64(rows))
	meansNeg := means.MultiplyConstant(-1.0)
	featuresTrans := X.Tranpose()
	vars := make([]float64, cols)
	for i := 0; i < cols; i++ {
		diff = featuresTrans.Rows[i].AddConstant(meansNeg.Values[i])
		vars[i] = (1.0 / (float64(rows) - 1.0)) * diff.ApplyFunc(func(n float64) float64 {
			return n * n
		}).Sum()
	}
	return means.Values, vars
}

func (m *NaiveBayes) classProbabilities(values *matrix.Vector) (probabilities map[ml.TargetValue]float64) {
	var product, variance float64
	probabilities = make(map[ml.TargetValue]float64)
	for i := range m.means {
		product = 1.0
		for j, mean := range m.means[i] {
			variance = m.vars[i][j]
			product = product * (1 / (math.Sqrt(2 * math.Pi * variance))) * math.Exp(-math.Pow(values.Values[j]-mean, 2.0)/(2*variance))
		}
		probabilities[i] = product
	}
	return
}

func (m *NaiveBayes) predictClass(probabilities map[ml.TargetValue]float64) (class ml.TargetValue) {
	maxProba := 0.0
	for target, probability := range probabilities {
		if probability > maxProba {
			class = target
			maxProba = probability
		}
	}
	return
}

func (m *NaiveBayes) Fit(data ml.DataSet) {
	m.means = make(map[ml.TargetValue][]float64)
	m.vars = make(map[ml.TargetValue][]float64)
	m.priorProba = make(map[ml.TargetValue]float64)
	rows, _ := data.Target.Dims()
	var X *matrix.DenseMatrix
	var meanTemp, varTemp []float64
	classIndicies := make(map[ml.TargetValue][]int)
	for rowIndex, targetVec := range data.Target.Rows {
		classIndicies[ml.TargetValue(targetVec.Values[0])] = append(classIndicies[ml.TargetValue(targetVec.Values[0])], rowIndex)
	}
	for targetVal, indicies := range classIndicies {
		X = matrix.GetSubSetByIndex(data.Features, indicies)
		meanTemp, varTemp = m.getMeanAndVariance(X)
		m.priorProba[targetVal] = float64(len(indicies)) / float64(rows)
		m.means[targetVal] = meanTemp
		m.vars[targetVal] = varTemp
	}
}

func (m *NaiveBayes) Predict(X *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, _ := X.Dims()
	res := matrix.InitializeMatrix(rows, 1)
	for rowIndex, rowVector := range X.Rows {
		res.Rows[rowIndex].Values[0] = float64(m.predictClass(m.classProbabilities(rowVector)))
	}
	return res
}
