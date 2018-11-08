from __future__ import print_function

import os
import pickle

from nltk.tag import StanfordPOSTagger
from nltk.tree import Tree

from distance_parser.helpers import *


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


class CTBCreator(object):
    '''Data path is assumed to be a directory with
       pkl files and a corpora subdirectory.
    '''

    def __init__(self,
                 wordembed_dim=300,
                 embeddingstd=0.1,
                 data_path=None,
                 tagger_path=None):
        assert data_path is not None
        assert tagger_path is not None
        dict_filepath = os.path.join(data_path, 'dict.pkl')
        data_filepath = os.path.join(data_path, 'parsed.pkl')
        train_filepath = os.path.join(data_path, "train.txt")
        valid_filepath = os.path.join(data_path, "dev.txt")
        test_filepath = os.path.join(data_path, "test.txt")

        self.st = StanfordPOSTagger(os.path.join(tagger_path, 'models/chinese-distsim.tagger'),
                                    os.path.join(tagger_path, 'stanford-postagger.jar'))

        print("building dictionary ...")
        f_dict = open(dict_filepath, 'wb')
        self.dictionary = Dictionary()

        print("loading trees from {}".format(train_filepath))
        train_trees = load_trees(train_filepath)
        print("loading trees from {}".format(valid_filepath))
        valid_trees = load_trees(valid_filepath)
        print("loading trees from {}".format(test_filepath))
        test_trees = load_trees(test_filepath)

        self.add_words(train_trees)
        self.dictionary.rebuild_by_freq()
        self.arc_dictionary = Dictionary()
        self.stag_dictionary = Dictionary()
        self.train = self.preprocess(train_trees, is_train=True)
        self.valid = self.preprocess(valid_trees, is_train=False)
        self.test = self.preprocess(test_trees, is_train=False)
        with open(dict_filepath, "wb") as file_dict:
            pickle.dump(self.dictionary, file_dict)
        with open(data_filepath, "wb") as file_data:
            pickle.dump((self.train, self.arc_dictionary,
                         self.stag_dictionary), file_data)
            pickle.dump(self.valid, file_data)
            pickle.dump(self.test, file_data)

        print(len(self.arc_dictionary.idx2word))
        print(self.arc_dictionary.idx2word)

    def add_words(self, trees):
        words, tags = [], []
        for tree in trees:
            tree = process_NONE(tree)
            words, tags = zip(*tree.pos())
            words = ['<s>'] + list(words) + ['</s>']
            for w in words:
                self.dictionary.add_word(w)

    def preprocess(self, parse_trees, is_train=False):
        sens_idx = []
        sens_tag = []
        sens_stag = []
        sens_arc = []
        distances = []
        sens = []
        trees = []

        print('\nConverting trees ...')
        for i, tree in enumerate(parse_trees):
            tree = process_NONE(tree)
            if i % 10 == 0:
                print("Done %d/%d\r" % (i, len(parse_trees)), end='')
            word_lexs, _ = zip(*tree.pos())
            idx = []
            for word in (['<s>'] + list(word_lexs) + ['</s>']):
                idx.append(self.dictionary[word])

            listerized_tree, arcs, tags = tree2list(tree)
            tags = ['<unk>'] + tags + ['<unk>']
            arcs = ['<unk>'] + arcs + ['<unk>']

            if type(listerized_tree) is str:
                listerized_tree = [listerized_tree]
            distances_sent, _ = distance(listerized_tree)
            distances_sent = [0] + distances_sent + [0]

            idx_arcs = []
            for arc in arcs:
                arc = precess_arc(arc)
                arc_id = self.arc_dictionary.add_word(arc) if is_train else self.arc_dictionary[arc]
                idx_arcs.append(arc_id)

            # the "tags" are the collapsed unary chains, i.e. FRAG+DT
            # at evaluation, we swap the word tag "DT" with the true tag in "stags" (see after)
            idx_tags = []
            for tag in tags:
                tag = precess_arc(tag)
                tag_id = self.arc_dictionary.add_word(tag) if is_train else self.arc_dictionary[tag]
                idx_tags.append(tag_id)

            assert len(distances_sent) == len(idx) - 1
            assert len(arcs) == len(idx) - 1
            assert len(idx) == len(word_lexs) + 2

            sens.append(word_lexs)
            trees.append(tree)
            sens_idx.append(idx)
            sens_tag.append(idx_tags)
            sens_arc.append(idx_arcs)
            distances.append(distances_sent)

        print('\nLabelling POS tags ...')
        st_outputs = self.st.tag_sents(sens)
        for i, word_tags in enumerate(st_outputs):
            if i % 10 == 0:
                print("Done %d/%d\r" % (i, len(parse_trees)), end='')
            word_tags = [t[1].split('#')[1] for t in word_tags]
            stags = ['<s>'] + list(word_tags) + ['</s>']

            # the "stags" are the original word tags included in the data files
            # we keep track of them so that, during evaluation, we can swap them with the original ones.
            idx_stags = []
            for stag in stags:
                stag_id = self.stag_dictionary.add_word(stag) if is_train else self.stag_dictionary[stag]
                idx_stags.append(stag_id)

            sens_stag.append(idx_stags)

        return sens_idx, sens_tag, sens_stag, \
               sens_arc, distances, sens, trees


if __name__ == '__main__':
    import sys

    CTBCreator(data_path=sys.argv[1], tagger_path=sys.argv[2])
