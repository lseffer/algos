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

func (m *KMeans) initializeCentroids(X *matrix.DenseMatrix) ([]*matrix.Vector, error) {
	_, cols := X.Dims()
	var err error
	var vec *matrix.Vector
	centroids := make([]*matrix.Vector, m.ClusterCount)
	for i := range centroids {
		vec, err = matrix.InitializeVector(cols)
		centroids[i] = vec
	}
	return centroids, err
}

// assignCentroids we initialize the centroids to a random row
// in the input matrix. Centroids can be assigned the same row.
func (m *KMeans) assignCentroids(X *matrix.DenseMatrix) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	rows, _ := X.Dims()
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

func (m *KMeans) updateCentroids(X *matrix.DenseMatrix, closestCentroids *matrix.Vector, currentCentroids []*matrix.Vector) ([]*matrix.Vector, error) {
	var err error
	var vec *matrix.Vector
	var closestCentroid int
	var newCentroids []*matrix.Vector
	centroidDivisor := make(map[int]float64)
	newCentroids, err = m.initializeCentroids(X)
	for i, rowVector := range X.Rows {
		closestCentroid = int(closestCentroids.Values[i])
		vec, err = rowVector.Add(newCentroids[closestCentroid])
		newCentroids[closestCentroid] = vec
		centroidDivisor[closestCentroid]++
	}
	for i, centroid := range newCentroids {
		vec, err = centroid.MultiplyConstant(1.0 / centroidDivisor[i])
		newCentroids[i] = vec
	}
	return newCentroids, err
}

// Fit the Kmeans model using the naive algorithm
func (m *KMeans) Fit(X *matrix.DenseMatrix) {
	var currentCentroids, newCentroids []*matrix.Vector
	var closestCentroids *matrix.Vector
	currentCentroids, _ = m.initializeCentroids(X)
	m.centroids = currentCentroids
	m.assignCentroids(X)
	for i := 0; i < m.MaxIterations; i++ {
		closestCentroids, _ = m.closestCentroidByRow(X)
		currentCentroids = m.GetCentroids()
		newCentroids, _ = m.updateCentroids(X, closestCentroids, currentCentroids)
		m.centroids = newCentroids
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
