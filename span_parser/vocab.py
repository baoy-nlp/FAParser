from __future__ import division
from __future__ import print_function

import json
import sys
from collections import defaultdict, OrderedDict

import numpy as np

from span_parser.parser import Parser
from span_parser.phrase_tree import PhraseTree


class Vocab(object):
    """
    Maps words, tags, and label actions to indices.
    """

    UNK = '<UNK>'
    START = '<s>'
    STOP = '</s>'

    def __init__(self, tree_file, verbose=True):

        if tree_file is not None:
            data = Vocab.vocab_init(
                tree_file=tree_file,
                verbose=verbose,
            )
            self.wdict = data['wdict']
            self.word_freq = data['word_freq']
            self.tdict = data['tdict']
            self.ldict = data['ldict']
            self.word_freq_list = []

            for word in self.wdict.keys():
                if word in self.word_freq:
                    self.word_freq_list.append(self.word_freq[word])
                else:
                    self.word_freq_list.append(0)
            self.init_tag()

    def init_tag(self):
        self.tag_list = [0 for _ in range(len(self.tdict))]
        for k, v in self.tdict.items():
            self.tag_list[v] = k
        self.label_seqs = []
        for label in self.ldict.keys():
            self.label_seqs.append(label)

    @staticmethod
    def vocab_init(tree_file, verbose=True):
        """
        Learn vocabulary from file of strings.
        """
        word_freq = defaultdict(int)
        tag_freq = defaultdict(int)
        label_freq = defaultdict(int)

        trees = PhraseTree.load_trees(tree_file)

        for i, tree in enumerate(trees):
            for (word, tag) in tree.sentence:
                word_freq[word] += 1
                tag_freq[tag] += 1

            for action in Parser.gold_actions(tree):
                if action.startswith('label-'):
                    label = action[6:]
                    label_freq[label] += 1

            if verbose:
                print('\rTree {}'.format(i), end='')
                sys.stdout.flush()

        if verbose:
            print('\r', end='')

        words = [
                    Vocab.UNK,
                    Vocab.START,
                    Vocab.STOP,
                ] + sorted(word_freq)
        wdict = OrderedDict((w, i) for (i, w) in enumerate(words))

        tags = [
                   Vocab.UNK,
                   Vocab.START,
                   Vocab.STOP,
               ] + sorted(tag_freq)
        tdict = OrderedDict((t, i) for (i, t) in enumerate(tags))

        labels = sorted(label_freq)
        ldict = OrderedDict((l, i) for (i, l) in enumerate(labels))

        if verbose:
            print('Loading features from {}'.format(tree_file))
            print('({} words, {} tags, {} nonterminal-chains)'.format(
                len(wdict),
                len(tdict),
                len(ldict),
            ))

        return {
            'wdict': wdict,
            'word_freq': word_freq,
            'tdict': tdict,
            'ldict': ldict,
        }

    @staticmethod
    def from_dict(data):
        new = Vocab(None)
        new.wdict = data['wdict']
        new.word_freq = data['word_freq']
        new.tdict = data['tdict']
        new.ldict = data['ldict']
        new.word_freq_list = data['word_freq_list']
        new.init_tag()
        return new

    def as_dict(self):
        return {
            'wdict': self.wdict,
            'word_freq': self.word_freq,
            'tdict': self.tdict,
            'ldict': self.ldict,
            'word_freq_list': self.word_freq_list
        }

    def save_json(self, filename):
        with open(filename, 'w') as fh:
            json.dump(self.as_dict(), fh)

    @staticmethod
    def load_json(filename):
        print('load vocab ...')
        with open(filename) as fh:
            data = json.load(fh, object_pairs_hook=OrderedDict)
        return Vocab.from_dict(data)

    def total_words(self):
        return len(self.wdict)

    def total_tags(self):
        return len(self.tdict)

    def total_label_actions(self):
        return 1 + len(self.ldict)

    def tag_id(self, tag_str):
        return self.tdict[tag_str] if tag_str in self.tdict else self.tdict[Vocab.UNK]

    def tag_str(self, id):
        return self.tag_list[id]

    def s_action_index(self, action):
        if action == 'sh':
            return 0
        elif action == 'comb':
            return 1
        else:
            raise ValueError('Not s-action: {}'.format(action))

    def l_action_index(self, action):
        if action == 'none':
            return 0
        elif action.startswith('label-'):
            label = action[6:]
            label_index = self.ldict.get(label, None)
            if label_index is not None:
                return 1 + label_index
            else:
                return 0
        else:
            raise ValueError('Not l-action: {}'.format(action))

    def s_action(self, index):
        return ('sh', 'comb')[index]

    def l_action(self, index):
        if index == 0:
            return 'none'
        else:
            # return 'label-' + self.ldict.keys()[index - 1]
            return 'label-' + self.label_seqs[index - 1]

    def index_sentences(self, sentence):
        """
        Array of indices for words and tags.
        """
        sentence = (
            [(Vocab.START, Vocab.START)] +
            sentence +
            [(Vocab.STOP, Vocab.STOP)]
        )

        # words = [
        #     self.wdict[w.decode('utf-8')]
        #     if w.decode('utf-8') in self.wdict else self.wdict[FeatureMapper.UNK]
        #     for (w, t) in sentence
        # ]
        words = [
            self.wdict[w]
            if w in self.wdict else self.wdict[Vocab.UNK]
            for (w, t) in sentence
        ]
        tags = [
            self.tdict[t]
            if t in self.tdict else self.tdict[Vocab.UNK]
            for (w, t) in sentence
        ]

        w = np.array(words).astype('int64')
        t = np.array(tags).astype('int64')

        return w, t

    def gold_data(self, reftree):
        """
        Static oracle for tree.
        """

        w, t = self.index_sentences(reftree.sentence)
        return {
            'tree': reftree,
            'w': w,
            't': t,
        }

    def gold_data_from_file(self, fname):
        """
        Static oracle for file.
        """
        trees = PhraseTree.load_trees(fname)
        result = []
        for tree in trees:
            sentence_data = self.gold_data(tree)
            result.append(sentence_data)
        return result
