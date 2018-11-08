from torchtext.vocab import Vocab
from collections import Counter


class TargetVocab(Vocab):
    def __init__(self,
                 label,
                 right,
                 tag,
                 max_size=None,
                 min_freq=1,
                 specials=['<pad>'],
                 vectors=None,
                 ):
        counter = Counter()
        index = 1
        for item in tag:
            counter.update([item] * index)
            index += 1
        for item in right:
            counter.update([item] * index)
            index += 1
        for item in label:
            counter.update([item] * index)
            index += 1
        super(TargetVocab, self).__init__(counter, max_size, min_freq, specials, vectors)
        self.T = len(specials)
        self.M = len(label) + self.T
        self.L = self.M + len(right)
        self.E = -1

    def index_sentence(self, words, unk="<unk>"):
        return [self.stoi[token] if token in self.stoi else
                self.stoi[unk] for token in words]
