import json
from typing import List, AnyStr

from .bpe import Bpe


class Vocabulary(object):
    def __new__(cls, vocab_type, dict_path, max_n_words, *args, **kwargs):

        if vocab_type == "word":
            return _Word(dict_path, max_n_words, *args, **kwargs)
        elif vocab_type == "bpe":
            return _BPE(dict_path, max_n_words, *args, **kwargs)
        elif vocab_type == "char":
            return _Char(dict_path, max_n_words, *args, **kwargs)
        else:
            print("Unknow vocabulary type {0}".format(vocab_type))
            raise ValueError


class _Vocabulary(object):
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 3

    def __init__(self, dict_path, max_n_words, *args, **kwargs):

        self._max_n_words = max_n_words
        self._dict_path = dict_path
        self._load_vocab(self._dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])

    def _init_dict(self):

        return {
            "<PAD>": (self.PAD, 0),
            "<UNK>": (self.UNK, 0),
            "<EOS>": (self.EOS, 0),
            "<BOS>": (self.BOS, 0)
        }

    def _load_vocab(self, path: str):

        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path) as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2token(self, word_id):

        return self._id2token[word_id]

    def tokenize(self, line: str):
        raise NotImplementedError

    def detokenize(self, tokens: List[AnyStr]):
        raise NotImplementedError


class _Word(_Vocabulary):
    def tokenize(self, line: str):
        return line.strip().split()

    def detokenize(self, tokens: List[AnyStr]):
        return ' '.join(tokens)


class _BPE(_Vocabulary):
    def __init__(self, dict_path, max_n_words, bpe_codes, *args, **kwargs):
        super(_BPE, self).__init__(dict_path, max_n_words, *args, **kwargs)

        self._bpe = Bpe(codes=bpe_codes)

    def tokenize(self, line: str):
        line = line.strip().split()

        return sum([self._bpe.segment_word(w) for w in line], [])

    def detokenize(self, tokens: List[AnyStr]):
        return ' '.join(tokens).replace("@@ ", "")


class _Char(_Vocabulary):
    def tokenize(self, line: str):
        return list(line.strip())

    def detokenize(self, tokens: List[AnyStr]):
        return ' '.join(tokens)


PAD = _Vocabulary.PAD
EOS = _Vocabulary.EOS
BOS = _Vocabulary.BOS
UNK = _Vocabulary.UNK
