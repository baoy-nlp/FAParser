from seq2seq_parser.utils.tree_analysis import eval_file

__all__ = [
    'Inner F1 Scorer'
]


class F1Scorer(object):
    def __init__(self, reference_path, digits_only=True, lc=False):
        self.ref_path = reference_path
        self.digits_only = digits_only
        self.lc = lc

    def corpus_f1(self, pred_file):
        accuracy, error = eval_file(pred_file=pred_file, gold_file=self.ref_path)
        print("error count:{}".format(error))

        if self.digits_only:
            accuracy = accuracy.fscore()
        return accuracy
