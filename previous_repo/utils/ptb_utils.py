from __future__ import print_function

import nltk
from nltk.tree import Tree

from NJUParser.dataset.dictionary import Dictionary


def load_trees(path, strip_top=True, strip_spmrl_features=True):
    trees = []
    with open(path) as infile:
        for line in infile:
            trees.append(Tree.fromstring(line))

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label() in ("TOP", "ROOT"):
                assert len(tree) == 1
                trees[i] = tree[0]
    return trees


class PTBLoader(object):
    def __init__(self, data_path, use_ext_embed=False):
        nltk.data.path.append(data_path)
        dict_path = os.path.join(data_path, "dict.pkl")
        ptb_path = os.path.join(data_path, "parsed.pkl")

        self.dictionary = Dictionary.load(dict_path)

