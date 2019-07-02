import os
import pickle

import nltk

from NJUParser.dataset.dictionary import Dictionary
from NJUParser.dataset.vocab import Vocab
from NJUParser.modules.tensor_ops import get_long_tensor


class TreeLoader(object):
    def __init__(self, data_path=None):
        assert data_path is not None
        nltk.data.path.append(data_path)
        dict_file = os.path.join(data_path, '{}.dict.pkl')
        data_file = os.path.join(data_path, 'parsed.pkl')

        print("loading dictionary ...")
        self.word_dict = Dictionary.load(dict_file.format("word"))
        self.arc_dict = Dictionary.load(dict_file.format("arc"))
        self.stag_dict = Dictionary.load(dict_file.format("stag"))
        self.vocab = Vocab.load(dict_file.format('vocab'))

        # build tree and distance
        print("loading tree and distance ...")
        with open(data_file, 'rb') as file_data:
            self.train = pickle.load(file_data)
            self.valid = pickle.load(file_data)
            self.test = pickle.load(file_data)

    @staticmethod
    def prepare_data(raw_data, batch_size):
        """
        :param raw_data: [words],[tags]
        :param batch_size:
        :return:
        """
        pass

    def batchify(self, data_name, batch_size):
        sents, trees = None, None
        if data_name == 'train':
            words, tags, stags, arcs, distances, sents, trees = self.train
        elif data_name == 'valid':
            words, tags, stags, arcs, distances, _, _ = self.valid
        elif data_name == 'test':
            words, tags, stags, arcs, distances, _, _ = self.test
        else:
            raise RuntimeError('need a correct data name')

        assert len(words) == len(distances)
        assert len(words) == len(tags)

        batch_words, batch_tags, batch_stags, batch_arcs, batch_dsts, = [], [], [], [], []
        batch_sents, batch_trees = [], []
        for i in range(0, len(words), batch_size):
            if i + batch_size >= len(words):
                continue

            if sents is not None:
                batch_sents.append(sents[i: i + batch_size])
                batch_trees.append(trees[i: i + batch_size])

            extracted_idxs = words[i: i + batch_size]
            extracted_tags = tags[i: i + batch_size]
            extracted_stags = stags[i: i + batch_size]

            extracted_arcs = arcs[i: i + batch_size]
            extracted_dsts = distances[i: i + batch_size]

            longest_idx = max([len(i) for i in extracted_idxs])
            longest_arc = longest_idx - 1

            mini_words, mini_tags, mini_stags, mini_arcs, mini_dsts, = [], [], [], [], []
            for idx, tag, stag, arc, dst in zip(extracted_idxs, extracted_tags, extracted_stags,
                                                extracted_arcs, extracted_dsts):
                padded_idx = idx + [-1] * (longest_idx - len(idx))
                padded_tag = tag + [0] * (longest_idx - len(tag))
                padded_stag = stag + [0] * (longest_idx - len(stag))

                padded_arc = arc + [0] * (longest_arc - len(arc))
                padded_dst = dst + [0] * (longest_arc - len(dst))

                mini_words.append(padded_idx)
                mini_tags.append(padded_tag)
                mini_stags.append(padded_stag)

                mini_arcs.append(padded_arc)
                mini_dsts.append(padded_dst)

            mini_words = get_long_tensor(mini_words)
            mini_tags = get_long_tensor(mini_tags)
            mini_stags = get_long_tensor(mini_stags)

            mini_arcs = get_long_tensor(mini_arcs)
            mini_dsts = get_long_tensor(mini_dsts)

            batch_words.append(mini_words)
            batch_tags.append(mini_tags)
            batch_stags.append(mini_stags)

            batch_arcs.append(mini_arcs)
            batch_dsts.append(mini_dsts)

        if sents is not None:
            return batch_words, batch_tags, batch_stags, batch_arcs, batch_dsts, batch_sents, batch_trees
        return batch_words, batch_tags, batch_stags, batch_arcs, batch_dsts
