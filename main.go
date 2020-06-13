package main

import (
	"algos/matrix"
	"algos/ml"
	"fmt"
	"math/rand"
	"time"
)

func makeRange(min, max int) []float64 {
	a := make([]float64, max-min+1)
	for i := range a {
		a[i] = float64(min + i)
	}
	return a
}

func makeRandRange(size int) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	a := make([]float64, size)
	for i := range a {
		a[i] = r.Float64() * 100.0
	}
	return a
}

func square(a float64) float64 {
	return a * a
}

func main() {
	var test1 = matrix.DenseMatrix{Rows: []*matrix.Vector{
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)}}}
	var test2 = matrix.DenseMatrix{Rows: []*matrix.Vector{
		&matrix.Vector{Values: makeRandRange(3)},
		&matrix.Vector{Values: makeRandRange(3)}}}
	// testm5T, _ := testm5.Tranpose()
	// fmt.Println(testm5.Dims(), testm5T)
	fmt.Println(test1)
	fmt.Println(test2)
	// fmt.Println(testm5.Multiply(testm5T))
	// fmt.Println(testm5.ApplyFunc(square))
	// fmt.Println(testm5.ReduceSum(0))
	var km = ml.KMeans{ClusterCount: 2, MaxIterations: 100, Tolerance: 0.1}
	km.Fit(&test1)
	fmt.Println(km.GetCentroids())
	fmt.Println(km.Predict(&test2))
}
