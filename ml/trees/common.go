package trees

import "algos/ml"

type treeNode struct {
	score         float64
	colIndex      int
	splitValue    float64
	depth         int
	majorityClass ml.ClassValue
	left          *treeNode
	right         *treeNode
}

type treeStack []*treeNode

func (s treeStack) Size() int {
	return len(s)
}

func (s treeStack) Push(v *treeNode) treeStack {
	return append(s, v)
}

func (s treeStack) Pop() (treeStack, *treeNode) {

	l := len(s)
	if l <= 0 {
		return nil, nil
	}
	return s[:l-1], s[l-1]
}
