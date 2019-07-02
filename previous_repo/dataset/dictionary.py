import pickle

import numpy


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [robust_id2word(s, vocab) for s in sents]
    else:
        return robust_id2word(sents, vocab)


def robust_id2word(sents, vocab):
    res = []
    for w in sents:
        if w == vocab.sos_id or w == vocab.pad_id:
            pass
        elif w == vocab.eos_id:
            break
        else:
            res.append(vocab.id2word[w])
    return res


class Dictionary(object):
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self, use_sos=False, use_eos=False, use_pad=False):
        self.word2id = {Dictionary.UNK_TOKEN: 0}
        self.id2word = [Dictionary.UNK_TOKEN]
        self.init_extra_token(use_sos, use_eos, use_pad)
        self.word2frq = {}

    def init_extra_token(self, sos=False, eos=False, pad=False):

        if sos:
            self.word2id[Dictionary.SOS_TOKEN] = 1
            self.id2word.append(Dictionary.SOS_TOKEN)
        if eos:
            self.word2id[Dictionary.EOS_TOKEN] = 2
            self.id2word.append(Dictionary.EOS_TOKEN)

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word.append(word)
            self.word2id[word] = len(self.id2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, item):
        if item in self.word2id:
            return self.word2id[item]
        else:
            return self.word2id[Dictionary.UNK_TOKEN]

    def rebuild_by_freq(self, thd=3, use_sos=False, use_eos=False, use_pad=False):
        """
        :param thd: int, freq threshold
        :param use_sos: bool
        :param use_eos: bool
        :param use_pad: bool
        :return:
        """
        self.word2id = {Dictionary.UNK_TOKEN: 0}
        self.id2word = [Dictionary.UNK_TOKEN]
        self.init_extra_token(use_sos, use_eos, use_pad)
        for k, v in self.word2frq.items():
            if v >= thd and (not k in self.id2word):
                self.id2word.append(k)
                self.word2id[k] = len(self.id2word) - 1

        print('Number of words:', len(self.id2word))
        return len(self.id2word)

    def class_weight(self):
        frq = [self.word2frq[self.id2word[i]] for i in range(len(self.id2word))]
        frq = numpy.array(frq).astype('float')
        weight = numpy.sqrt(frq.max() / frq)
        weight = numpy.clip(weight, a_min=0., a_max=5.)

        return weight

    def save(self, save_path):
        with open(save_path, "wb") as file_dict:
            pickle.dump(self, file_dict)

    @staticmethod
    def load(saved_path):
        return pickle.load(open(saved_path, "rb"))
