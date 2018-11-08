class FScore(object):
    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct  # correct brackets
        self.predcount = predcount  # total predicted brackets
        self.goldcount = goldcount  # total gold brackets

    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        fscore = self.fscore()
        return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision,
            recall,
            fscore,
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self

    def __add__(self, other):
        return FScore(self.correct + other.correct,
                      self.predcount + other.predcount,
                      self.goldcount + other.goldcount)

    def __cmp__(self, other):
        if self.fscore() < other.fscore():
            return -1
        elif self.fscore() == other.fscore():
            return 0
        else:
            return 1


class Accuracy(object):
    def __init__(self, correct=0, goldcount=0):
        self.correct = correct  # correct brackets
        self.goldcount = goldcount  # total gold brackets

    def precision(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        return 'P= {:0.2f}'.format(
            precision,
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.goldcount += other.goldcount
        return self

    def __add__(self, other):
        return FScore(self.correct + other.correct,
                      self.goldcount + other.goldcount)

    def __cmp__(self, other):
        return cmp(self.precision(), other.precision())
