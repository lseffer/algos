package trees

import (
	"algos/matrix"
	"algos/ml"
	"sync"
)

type splitResults struct {
	score      float64
	colIndex   int
	splitValue float64
	leftData   *matrix.DenseMatrix
	rightData  *matrix.DenseMatrix
}

type splitFinder interface {
	algorithm(X *matrix.DenseMatrix, criteria splitCriteria, rowsStart, rowsEnd int) (splitResults, error)
}

type greedySplitFinder struct{}

type concurrentSplitFinder struct {
	jobs int
	s    splitFinder
}

func (f concurrentSplitFinder) algorithm(X *matrix.DenseMatrix, criteria splitCriteria, rowsStart, rowsEnd int) (result splitResults, err error) {
	var jobs int
	var wg sync.WaitGroup
	rows, _ := X.Dims()
	if rows <= f.jobs {
		jobs = rows / 2
	}
	wg.Add(jobs)
	c := make(chan splitResults, jobs)
	chunkSize := (rows + jobs - 1) / jobs
	for i := 0; i < rows; i += chunkSize {
		end := i + chunkSize
		if end > rows {
			end = rows
		}
		go func(s, e int) {
			var res splitResults
			res, _ = f.s.algorithm(X, criteria, s, e)
			c <- res
			wg.Done()
		}(i, end)
	}
	wg.Wait()
	close(c)
	bestScore := 100.0
	for res := range c {
		if res.score < bestScore {
			bestScore = res.score
			result = res
		}
	}
	return result, nil
}

func (f greedySplitFinder) algorithm(X *matrix.DenseMatrix, criteria splitCriteria, rowsStart, rowsEnd int) (splitResults, error) {
	var err error
	var bestColIndex int
	var leftClassVector, rightClassVector ml.ClassVector
	var left, right, bestLeft, bestRight *matrix.DenseMatrix
	var score, bestScore, bestSplitValue float64
	_, cols := X.Dims()
	bestScore = 100.0
	for _, rowVector := range X.Rows[rowsStart:rowsEnd] {
		for colIndex, splitValue := range rowVector.Values[:cols-1] {
			left, right, err = matrix.SplitMatrix(X, colIndex, splitValue)
			leftClassVector, err = ml.NewClassVector(left)
			rightClassVector, err = ml.NewClassVector(right)
			score, err = scoreSplit(leftClassVector, rightClassVector, criteria)
			if score < bestScore {
				bestScore = score
				bestColIndex = colIndex
				bestSplitValue = splitValue
				bestLeft = left
				bestRight = right
			}
		}
	}
	return splitResults{score: bestScore, colIndex: bestColIndex, splitValue: bestSplitValue, leftData: bestLeft, rightData: bestRight}, err
}

func scoreSplit(left, right ml.ClassVector, criteria splitCriteria) (float64, error) {
	var err error
	var leftScore, rightScore float64
	leftRows, _ := left.Values.Dims()
	rightRows, _ := right.Values.Dims()
	totalRows := leftRows + rightRows
	if totalRows <= 0.0 {
		return 0.0, err
	}
	leftScore, err = scoreClassVector(left, criteria)
	rightScore, err = scoreClassVector(right, criteria)
	totalScore := leftScore*float64(leftRows)/float64(totalRows) + rightScore*float64(rightRows)/float64(totalRows)
	return totalScore, err
}
