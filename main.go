package main

import (
	"algos/matrix"
	"fmt"
)

func makeRange(min, max int) []float32 {
	a := make([]float32, max-min+1)
	for i := range a {
		a[i] = float32(min + i)
	}
	return a
}

func main() {
	var testm5 = matrix.DenseMatrix{Rows: [][]float32{makeRange(1, 3), makeRange(1, 3), makeRange(1, 3)}}
	testm5T, _ := testm5.Tranpose()
	fmt.Println(testm5.Dims(), testm5T)
	fmt.Println(testm5)
	fmt.Println(testm5.Multiply(testm5T))
}
