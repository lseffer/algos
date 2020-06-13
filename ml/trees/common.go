package trees

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
