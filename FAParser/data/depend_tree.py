from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DependTree(object):
    def __init__(self, leaf, sentence):
        super(DependTree, self).__init__()
        self.leaf = leaf
        self.sentence = sentence
        self._head = None
        self._head_rel = None
        self._children = None
        self._child_rels = None

    @property
    def children(self):
        return self._children, self._child_rels

    @property
    def head(self):
        return self._head, self._head_rel


class BottomUpDepTree(DependTree):
    def __init__(self, head, rel, leaf, sentence=None):
        super(BottomUpDepTree, self).__init__(leaf, sentence)
        self._head = head
        self._head_rel = rel


class TopDownDepTree(DependTree):
    def __init__(self, children, rels, leaf, sentence=None):
        super(TopDownDepTree, self).__init__(leaf, sentence)
        self._children = children
        self._child_rels = rels
