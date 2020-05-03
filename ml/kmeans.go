package ml

import (
	"algos/matrix"
	"math"
	"math/rand"
	"time"
)

// KMeans ml model
type KMeans struct {
	ClusterCount  int
	MaxIterations int
	Tolerance     float64
	centroids     []*matrix.Vector
}

// initializeCentroids we initialize the centroids to a random row
// in the input matrix. Centroids can be assigned the same row.
func (m *KMeans) initializeCentroids(X *matrix.DenseMatrix) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rows, _ := X.Dims()
	m.centroids = make([]*matrix.Vector, m.ClusterCount)
	for i := 0; i < m.ClusterCount; i++ {
		randomRowIndex := r.Intn(rows)
		m.centroids[i] = X.Rows[randomRowIndex]
	}
}

func (m *KMeans) closestCentroid(vec *matrix.Vector) (int, error) {
	var centroidDistance float64
	var closest int
	var err error
	distance := math.Inf(0)
	for i, centroid := range m.centroids {
		centroidDistance, err = vec.EuclideanDistance(centroid)
		if centroidDistance < distance {
			closest = i
			distance = centroidDistance
		}
	}
	return closest, err
}

func (m *KMeans) closestCentroidByRow(X *matrix.DenseMatrix) (*matrix.Vector, error) {
	rows, _ := X.Dims()
	var err error
	var result *matrix.Vector
	result, err = matrix.InitializeVector(rows)
	var closest int
	for i, rowVector := range X.Rows {
		closest, err = m.closestCentroid(rowVector)
		result.Values[i] = float64(closest)
	}
	return result, err
}

func (m *KMeans) updateCentroids(X *matrix.DenseMatrix, closestCentroids *matrix.Vector) error {
	_, cols := X.Dims()
	centroidPartitionLengths := make(map[int]int)
	var closestCentroid int
	var centroidPartition *matrix.DenseMatrix
	var partitionMatrix *matrix.DenseMatrix
	var centroidPartitionReduced *matrix.Vector
	centroidPartitionMap := make(map[int]*matrix.DenseMatrix)
	var newCentroid *matrix.Vector
	var partitionRows int
	var err error
	for i := range X.Rows {
		closestCentroid = int(closestCentroids.Values[i])
		centroidPartitionLengths[closestCentroid]++
	}
	for centroidIndex, partitionRows := range centroidPartitionLengths {
		partitionMatrix, err = matrix.InitializeMatrix(partitionRows, cols)
		centroidPartitionMap[centroidIndex] = partitionMatrix
	}
	for i, rowVector := range X.Rows {
		closestCentroid = int(closestCentroids.Values[i])
		centroidPartition = centroidPartitionMap[closestCentroid]
		centroidPartition.Rows = append(centroidPartition.Rows, rowVector)
	}
	for centroidIndex, centroidPartition := range centroidPartitionMap {
		centroidPartition = centroidPartitionMap[centroidIndex]
		partitionRows, _ = centroidPartition.Dims()
		centroidPartitionReduced, err = centroidPartition.ReduceSum(1)
		newCentroid, err = centroidPartitionReduced.MultiplyConstant(1.0 / float64(partitionRows))
		m.centroids[centroidIndex] = newCentroid
	}
	return err
}

// Fit the Kmeans model using the naive algorithm
func (m *KMeans) Fit(X *matrix.DenseMatrix) {
	m.initializeCentroids(X)
	var closestCentroids *matrix.Vector
	for i := 0; i < m.MaxIterations; i++ {
		closestCentroids, _ = m.closestCentroidByRow(X)
		m.updateCentroids(X, closestCentroids)
	}
}

// GetCentroids return the current centroids
func (m *KMeans) GetCentroids() []*matrix.Vector {
	return m.centroids
}

// Predict on data using the fitted centroids
func (m *KMeans) Predict(X *matrix.DenseMatrix) (*matrix.Vector, error) {
	return m.closestCentroidByRow(X)
}
