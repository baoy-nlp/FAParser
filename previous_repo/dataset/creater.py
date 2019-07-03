from __future__ import print_function

import os
import pickle

from nltk.tree import Tree

from FAParser.dataset.dictionary import Dictionary
from FAParser.dataset.helper import *
from FAParser.dataset.vocab import Vocab


def load_trees(path, strip_top=True):
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


class TreeCreator(object):
    def __init__(self, output_path=None, treebank_path=None):
        """
        :param output_path:
        :param treebank_path:
        """
        train_filepath = os.path.join(treebank_path, "train.clean")
        valid_filepath = os.path.join(treebank_path, "dev.clean")
        test_filepath = os.path.join(treebank_path, "test.clean")

        dict_filepath = os.path.join(output_path, '{}.dict.pkl')
        data_filepath = os.path.join(output_path, 'parsed.pkl')

        self.word_dict = Dictionary()

        print("loading trees from {}".format(train_filepath))
        train_trees = load_trees(train_filepath)
        print("loading trees from {}".format(valid_filepath))
        valid_trees = load_trees(valid_filepath)
        print("loading trees from {}".format(test_filepath))
        test_trees = load_trees(test_filepath)

        self.construct_vocab(train_trees)
        self.word_dict.rebuild_by_freq()
        self.arc_dict = Dictionary()
        self.stag_dict = Dictionary()

        self.train = self.pre_construct(train_trees, is_train=True)
        self.valid = self.pre_construct(valid_trees, is_train=False)
        self.test = self.pre_construct(test_trees, is_train=False)

        self.vocab = Vocab(word=self.word_dict, tag=self.stag_dict, arc=self.arc_dict)
        self.vocab.save(vocab_file=dict_filepath.format("vocab"))
        self.word_dict.save(save_path=dict_filepath.format("word"))
        self.arc_dict.save(save_path=dict_filepath.format("arc"))
        self.stag_dict.save(save_path=dict_filepath.format("stag"))
        with open(data_filepath, "wb") as file_data:
            pickle.dump(self.train, file_data)
            pickle.dump(self.valid, file_data)
            pickle.dump(self.test, file_data)

        print(len(self.arc_dict.id2word))
        print(self.arc_dict.id2word)

    def construct_vocab(self, trees):
        for tree in trees:
            words, tags = zip(*tree.pos())
            words = ['<s>'] + list(words) + ['</s>']
            for w in words:
                self.word_dict.add_word(w)

    def pre_construct(self, parse_trees, is_train=False):
        sens_words = []
        sens_tags = []
        sens_ptags = []
        sens_arcs = []
        sens_dsts = []
        sens = []
        trees = []

        print('\nConverting trees ...')
        for i, tree in enumerate(parse_trees):
            if i % 10 == 0:
                print("Done %d/%d\r" % (i, len(parse_trees)), end='')
            word_lexs, wtags = zip(*tree.pos())
            word_idx = []
            for word in (['<s>'] + list(word_lexs) + ['</s>']):
                word_idx.append(self.word_dict[word])

            list_tree, arcs, tags = tree2list(tree)
            stags = ['<s>'] + list(wtags) + ['</s>']
            tags = ['<unk>'] + tags + ['<unk>']
            arcs = ['<unk>'] + arcs + ['<unk>']

            if type(list_tree) is str:
                list_tree = [list_tree]
            distance, _ = get_distance(list_tree)
            distance = [0] + distance + [0]

            idx_arcs = []
            for arc in arcs:
                arc = precess_arc(arc)
                arc_id = self.arc_dict.add_word(arc) if is_train else self.arc_dict[arc]
                idx_arcs.append(arc_id)

            idx_stags = []
            for stag in stags:
                stag_id = self.stag_dict.add_word(stag) if is_train else self.stag_dict[stag]
                idx_stags.append(stag_id)

            idx_tags = []
            for tag in tags:
                tag = precess_arc(tag)
                tag_id = self.arc_dict.add_word(tag) if is_train else self.arc_dict[tag]
                idx_tags.append(tag_id)

            assert len(distance) == len(word_idx) - 1
            assert len(arcs) == len(word_idx) - 1
            assert len(word_idx) == len(word_lexs) + 2
            assert len(stags) == len(tags)

            sens.append(word_lexs)
            trees.append(tree)
            sens_words.append(word_idx)
            sens_tags.append(idx_tags)
            sens_arcs.append(idx_arcs)
            sens_ptags.append(idx_stags)
            sens_dsts.append(distance)

        return sens_words, sens_tags, sens_ptags, sens_arcs, sens_dsts, sens, trees
