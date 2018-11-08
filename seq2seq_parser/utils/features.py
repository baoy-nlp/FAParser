from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from collections import defaultdict, OrderedDict

from seq2seq_parser.utils.phrase_tree import PhraseTree


class FeatureMapper(object):
    """
    Maps words, tags, and label actions to indices.
    """

    @staticmethod
    def vocab_init(fname, verbose=True):
        """
        Learn vocabulary from file of strings.
        """
        tag_freq = defaultdict(int)

        trees = PhraseTree.load_treefile(fname)

        for i, tree in enumerate(trees):
            for (word, tag) in tree.sentence:
                tag_freq[tag] += 1

            if verbose:
                print('\rTree {}'.format(i), end='')
                sys.stdout.flush()

        if verbose:
            print('\r', end='')

        tags = ['XX'] + sorted(tag_freq)
        tdict = OrderedDict((t, i) for (i, t) in enumerate(tags))

        if verbose:
            print('Loading features from {}'.format(fname))
            print('( {} tags)'.format(
                len(tdict),
            ))

        return {
            'tdict': tdict,
        }

    def __init__(self, vocabfile, verbose=True):

        if vocabfile is not None:
            data = FeatureMapper.vocab_init(
                fname=vocabfile,
                verbose=verbose,
            )
            self.tdict = data['tdict']

            self.init_tag()

    def init_tag(self):
        self.tag_list = [0 for _ in range(len(self.tdict))]
        for k, v in self.tdict.items():
            self.tag_list[v] = k

    @staticmethod
    def from_dict(data):
        new = FeatureMapper(None)
        new.tdict = data['tdict']
        new.init_tag()
        return new

    def as_dict(self):
        return {
            'tdict': self.tdict,
        }

    def save_json(self, filename):
        with open(filename, 'w') as fh:
            json.dump(self.as_dict(), fh)

    @staticmethod
    def load_json(filename):
        print('load vocab ...')
        with open(filename) as fh:
            data = json.load(fh, object_pairs_hook=OrderedDict)
        return FeatureMapper.from_dict(data)

    def tag_id(self, tag_str):
        return self.tdict[tag_str] if tag_str in self.tdict else self.tdict[FeatureMapper.UNK]

    def tag_str(self, id):
        return self.tag_list[id]

    def is_tag(self, tag_str):
        return tag_str in self.tdict

    def seq_to_tree(self, translate):
        pad_endp = 0
        rm_brackets = 0
        fix_brackets = 0
        stack = []
        word_format = "{})"
        for item in translate:
            if not item.startswith("/"):
                stack.append([item, self.is_tag(item), False])
            else:
                query = item[1:]
                item_str = ")"
                could_find = False
                has_find = False
                while len(stack) > 0:
                    key = stack.pop(-1)
                    if not key[2] and not key[1]:  # mean is a open symbol
                        if not could_find:
                            item_str = key[0]
                        else:
                            has_find = True
                            item_str = "({} {}".format(key[0], item_str)
                            if key[0] != query:
                                fix_brackets += 1
                        break
                    else:  # mean is a mid phrase
                        could_find = True
                        if not key[2]:
                            item_str = "({} {} {}".format(key[0], word_format, item_str)
                        else:
                            item_str = "{} {}".format(key[0], item_str)
                if not has_find:
                    rm_brackets += 1

                if not could_find:
                    rm_brackets += 1

                stack.append((item_str, False, False))

        if len(stack) > 1:
            pad_endp += 1
        res_list = [stack for item in stack]


if __name__ == "__main__":
    outfile = "../data/vocab.json"
    import sys

    infile = sys.argv[1]

    fm = FeatureMapper(infile)
    fm.save_json(outfile)

    print("init the vocab finish")
