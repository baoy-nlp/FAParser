#!/usr/bin/env python

"""
Recursive representation of a phrase-structure parse tree
    for natural language sentences.

Implement based on Span-Parser


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict


class PhraseTree(object):
    puncs = [",", ".", ":", "``", "''", "PU"]  # (COLLINS.prm)

    def __init__(self, symbol=None, children=None, sentence=None, leaf=None):
        self.symbol = symbol  # label at top node
        self.children = children  # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf  # word at bottom level else None

        self._str = None

    def __str__(self):
        if self._str is None:
            if self.leaf is None:
                child_str = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, child_str)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1],
                    self.sentence[self.leaf][0],
                )
        return self._str

    def propagate_sentence(self, sentence):
        """
        Recursively assigns sentence (list of (word, POS) pairs)
            to all nodes of a tree.
        """
        self.sentence = sentence
        if self.children is not None:
            for child in self.children:
                child.propagate_sentence(sentence)

    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                self.symbol,
                self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result

    @property
    def left_span(self):
        if self.leaf is not None:
            return self.leaf
        else:
            return self.children[0].left_span

    @property
    def right_span(self):
        if self.leaf is not None:
            return self.leaf
        else:
            return self.children[-1].right_span

    def brackets(self, advp_prt=True, counts=None):

        if counts is None:
            counts = defaultdict(int)

        if self.leaf is not None:
            return {}

        nonterm = self.symbol
        if advp_prt and nonterm == 'PRT':
            nonterm = 'ADVP'

        left = self.left_span()
        right = self.right_span()

        # ignore punctuation
        while (
                        left < len(self.sentence) and
                        self.sentence[left][1] in PhraseTree.puncs
        ):
            left += 1
        while (
                        right > 0 and self.sentence[right][1] in PhraseTree.puncs
        ):
            right -= 1

        if left <= right and nonterm != 'TOP':
            counts[(nonterm, left, right)] += 1

        for child in self.children:
            child.brackets(advp_prt=advp_prt, counts=counts)

        return counts

    @property
    def phrase(self):
        if self.leaf is not None:
            return [(self.leaf, self.symbol)]
        else:
            result = []
            for child in self.children:
                result.extend(child.phrase)
            return result

    def compare(self, gold, advp_prt=True):
        """
        returns (Precision, Recall, F-measure)
        """
        pred_brackets = self.brackets(advp_prt)
        gold_brackets = gold.brackets(advp_prt)

        correct = 0
        for gb in gold_brackets:
            if gb in pred_brackets:
                correct += min(gold_brackets[gb], pred_brackets[gb])

        pred_total = sum(pred_brackets.values())
        gold_total = sum(gold_brackets.values())

        return correct, pred_total, gold_total

    def enclosing(self, i, j):
        """
        Returns the left and right indices of the labeled span in the tree
            which is next-larger than (i, j)
            (whether or not (i, j) is itself a labeled span)
        """
        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                if (left == i) and (right == j):
                    break
                return child.enclosing(i, j)

        return tuple([self.left_span, self.right_span])

    def span_labels(self, i, j):
        """
        Returns a list of span labels (if any) for (i, j)
        """
        if self.leaf is not None:
            return []

        if (self.left_span() == i) and (self.right_span() == j):
            result = [self.symbol]
        else:
            result = []

        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                result.extend(child.span_labels(i, j))
                break

        return result

    def rotate_tree(self):
        if self.leaf:
            return
        for child in self.children:
            child.rotate_tree()
        self.children = self.children[::-1]


def parse(line, index, sentence):
    """((...) (...) w/t (...)). returns pos and tree, and carries sent out."""

    assert line[index] == '(', "Invalid tree string {} at {}".format(line, index)
    index += 1
    symbol = None
    children = []
    leaf = None
    while line[index] != ')':
        if line[index] == '(':
            index, t = parse(line, index, sentence)
            children.append(t)

        else:
            if symbol is None:
                # symbol is here!
                rpos = min(line.find(' ', index), line.find(')', index))
                # see above N.B. (find could return -1)

                symbol = line[index:rpos]  # (word, tag) string pair

                index = rpos
            else:
                rpos = line.find(')', index)
                word = line[index:rpos]
                sentence.append((word, symbol))
                leaf = len(sentence) - 1
                index = rpos

        if line[index] == " ":
            index += 1

    assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

    t = PhraseTree(
        symbol=symbol,
        children=children,
        sentence=sentence,
        leaf=leaf,
    )

    return (index + 1), t


def str2tree(line):
    """
    Loads a tree from a tree in PTB parenthetical format.
    """
    line += " "
    sentence = []
    _, t = parse(line, 0, sentence)

    if t.symbol == 'TOP' and len(t.children) == 1:
        t = t.children[0]

    return t


def load_tree_file(fname):
    trees = []
    for line in open(fname):
        line = line.replace('S1', 'TOP')
        t = str2tree(line)
        trees.append(t)
    return trees


def load_raw2sents(fname):
    sents = []
    for line in open(fname):
        if line.strip() != '':
            sent = []
            word_tags = line.strip().split(' ')
            for word_tag in word_tags:
                items = word_tag.strip().split('_')
                sent.append((items[0], items[1]))
            sents.append(sent)
    return sents


def load_tree2raw(tree_file, sent_file):
    trees = load_tree_file(tree_file)

    def tree2sents(sentence):
        sent = ""
        for w, t in sentence:
            sent += (w + "_" + t) + " "
        return sent

    f = open(sent_file, 'w')
    for tree in trees:
        f.write(tree2sents(tree.sentence))
        f.write('\n')
    f.close()
