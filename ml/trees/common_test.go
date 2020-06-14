package trees

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTreeStack(t *testing.T) {
	rootNode := &treeNode{depth: 0}
	s := make(treeStack, 0)
	newStack, node := s.Pop()
	assert.Nil(t, newStack)
	assert.Nil(t, node)
	s = s.Push(rootNode)
	assert.Equal(t, 1, len(s))
	assert.Equal(t, 1, s.Size())
}
