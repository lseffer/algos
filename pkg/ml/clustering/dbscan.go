package ml

import (
	"github.com/lseffer/algos/pkg/matrix"
)

const noise = -1.0

// DBScan implements the popular DBScan clustering algorithm
type DBScan struct {
	NeighborMin   int
	Epsilon       float64
	clusterLabels map[int]int
}

// Fit the model
func (m *DBScan) Fit(X *matrix.DenseMatrix) {
	var neighbors []int
	m.clusterLabels = make(map[int]int)
	clusterCounter := 0
	for rowIdx := range X.Rows {
		if _, ok := m.clusterLabels[rowIdx]; ok {
			// Point already has a label
			continue
		}
		neighbors = m.rangeQuery(X, rowIdx)
		if len(neighbors) < m.NeighborMin {
			m.clusterLabels[rowIdx] = noise
			// Point now has a label, we can continue
			continue
		}
		clusterCounter++
		m.clusterLabels[rowIdx] = clusterCounter
		seedSet := make(map[int]bool)
		for _, neighbor := range neighbors {
			seedSet[neighbor] = true
		}
		seedSet[rowIdx] = true
		for seedRowIdx := range seedSet {
			if _, ok := m.clusterLabels[seedRowIdx]; ok {
				continue
			}
			m.clusterLabels[seedRowIdx] = clusterCounter
			neighbors = m.rangeQuery(X, seedRowIdx)
			if len(neighbors) > m.NeighborMin {
				for neighbor := range neighbors {
					seedSet[neighbor] = true
				}
			}
		}
	}
}

func (m *DBScan) rangeQuery(X *matrix.DenseMatrix, candidateRow int) (res []int) {
	for rowIdx, row := range X.Rows {
		if rowIdx != candidateRow {
			if dist, err := row.EuclideanDistance(X.Rows[candidateRow]); dist < m.Epsilon && err == nil {
				res = append(res, rowIdx)
			}
		}
	}
	return
}
