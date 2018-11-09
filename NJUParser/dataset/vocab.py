# coding=utf-8

from __future__ import print_function

import pickle
from collections import Counter
from itertools import chain


class VocabEntry(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2id = dict()

        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    @staticmethod
    def from_corpus(corpus, size=50000, freq_cutoff=0):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]

        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))

    @staticmethod
    def from_bin_file(file_path):
        import os
        if os.path.getsize(file_path) > 0:
            return pickle.load(open(file_path, 'rb'))
        else:
            raise RuntimeWarning("Empty vocab")


if __name__ == '__main__':
    raise NotImplementedError
