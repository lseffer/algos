package main

import (
	"log"

	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
	"github.com/lseffer/algos/pkg/ml/trees"
)

func main() {
	features, _ := matrix.ReadCsvFileToMatrix("/Users/leonardseffer/Downloads/random_regression_problem.csv")
	target, _ := matrix.ReadCsvFileToMatrix("/Users/leonardseffer/Downloads/random_regression_problem_y.csv")
	data, _ := ml.NewDataSet(features, target)
	model := trees.NewDecisionTreeRegressor(2, 50, trees.ConcurrentSplitFinder{Jobs: 100, SplitFinder: trees.GreedySplitFinder{}})
	log.Println("start training...")
	model.Fit(data)
}
