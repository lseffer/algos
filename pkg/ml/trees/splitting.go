package trees

import (
	"math"
	"sync"

	"github.com/lseffer/algos/pkg/matrix"
	"github.com/lseffer/algos/pkg/ml"
)

type splitResults struct {
	score      float64
	colIndex   int
	splitValue float64
	leftData   ml.DataSet
	rightData  ml.DataSet
}

type splitFinder interface {
	algorithm(data ml.DataSet, criteria splitCriteria, rowsStart, rowsEnd int) splitResults
}

// GreedySplitFinder finds the optimal split by iterating all columns and values and finding the split with lowest criteria score
type GreedySplitFinder struct{}

// ConcurrentSplitFinder makes any splitFinder run in parallel with goroutines. It also implements the splitFinder interface.
type ConcurrentSplitFinder struct {
	jobs int
	s    splitFinder
}

func (f ConcurrentSplitFinder) algorithm(data ml.DataSet, criteria splitCriteria, rowsStart, rowsEnd int) (result splitResults) {
	var jobs int
	var wg sync.WaitGroup
	rows, _ := data.Features.Dims()
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
			res = f.s.algorithm(data, criteria, s, e)
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
	return result
}

func colSplitFactory(colIndex int, splitValue float64) (res matrix.Qualifier) {
	res = func(rowIndex int, values *matrix.Vector) bool {
		return values.Values[colIndex] < splitValue
	}
	return
}

func (f GreedySplitFinder) algorithm(data ml.DataSet, criteria splitCriteria, rowsStart, rowsEnd int) (res splitResults) {
	var leftIndices, rightIndices []int
	var left, right ml.DataSet
	var score, bestScore float64
	bestScore = math.Inf(1)
	for _, rowVector := range data.Features.Rows[rowsStart:rowsEnd] {
		for colIndex, splitValue := range rowVector.Values {
			leftIndices, rightIndices = matrix.Split(data.Features, colSplitFactory(colIndex, splitValue))
			left.Features = matrix.GetSubSetByIndex(data.Features, leftIndices)
			left.Target = matrix.GetSubSetByIndex(data.Target, leftIndices)
			right.Features = matrix.GetSubSetByIndex(data.Features, rightIndices)
			right.Target = matrix.GetSubSetByIndex(data.Target, rightIndices)
			score = scoreSplit(left, right, criteria)
			if score < bestScore {
				res = splitResults{
					score:      score,
					colIndex:   colIndex,
					splitValue: splitValue,
					leftData:   left,
					rightData:  right,
				}
				bestScore = score
			}
		}
	}
	return res
}

func scoreSplit(left, right ml.DataSet, criteria splitCriteria) (score float64) {
	leftRows, _ := left.Features.Dims()
	rightRows, _ := right.Features.Dims()
	totalRows := leftRows + rightRows
	if totalRows <= 0.0 {
		return 0.0
	}
	leftScore := criteria.formula(left)
	rightScore := criteria.formula(right)
	score = leftScore*float64(leftRows)/float64(totalRows) + rightScore*float64(rightRows)/float64(totalRows)
	return score
}
