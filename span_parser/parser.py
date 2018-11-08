"""
Shift-Combine-Label parser.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from span_parser.phrase_tree import PhraseTree
from span_parser.measures import FScore


class Parser(object):
    def __init__(self, n):
        """
        Initial state for parsing an n-word sentence.
        """
        self.n = n
        self.i = 0
        self.stack = []

    def can_shift(self):
        return (self.i < self.n)

    def can_combine(self):
        return (len(self.stack) > 1)

    def shift(self):
        j = self.i  # (index of shifted word)
        treelet = PhraseTree(leaf=j)
        self.stack.append((j, j, [treelet]))
        self.i += 1

    def combine(self):
        (_, right, treelist0) = self.stack.pop()
        (left, _, treelist1) = self.stack.pop()
        self.stack.append((left, right, treelist1 + treelist0))

    def label(self, nonterminals=[]):

        for nt in nonterminals:
            (left, right, trees) = self.stack.pop()
            tree = PhraseTree(symbol=nt, children=trees)
            self.stack.append((left, right, [tree]))

    def take_action(self, action):
        if action == 'sh':
            self.shift()
        elif action == 'comb':
            self.combine()
        elif action == 'none':
            return
        elif action.startswith('label-'):
            self.label(action[6:].split('-'))
        else:
            raise RuntimeError('Invalid Action: {}'.format(action))

    def finished(self):
        return (
            (self.i == self.n) and
            (len(self.stack) == 1) and
            (len(self.stack[0][2]) == 1)
        )

    def tree(self):
        if not self.finished():
            raise RuntimeError('Not finished.')
        return self.stack[0][2][0]

    def s_features(self):
        """
        Features for predicting structural action (shift, combine):
            (pre-s1-span, s1-span, s0-span, post-s0-span)
        Note features use 1-based indexing:
            ... a span of (1, 1) means the first word of sentence
            ... (x, x-1) means no span
        """
        lefts = []
        rights = []

        # pre-s1-span
        lefts.append(1)
        if len(self.stack) < 2:
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            rights.append(s1_left - 1)

        # s1-span
        if len(self.stack) < 2:
            lefts.append(1)
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            lefts.append(s1_left)
            s1_right = self.stack[-2][1] + 1
            rights.append(s1_right)

        # s0-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def l_features(self):
        """
        Features for predicting label action:
            (pre-s0-span, s0-span, post-s0-span)
        """
        lefts = []
        rights = []

        # pre-s0-span
        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            rights.append(s0_left - 1)

        # s0-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def s_oracle(self, tree):
        """
        Returns correct structural action in current (arbitrary) state,
            given gold tree.
            Deterministic (prefer combine).
        """
        if not self.can_shift():
            return 'comb'
        elif not self.can_combine():
            return 'sh'
        else:
            (left0, right0, _) = self.stack[-1]
            a, _ = tree.enclosing(left0, right0)
            if a == left0:
                return 'sh'
            else:
                return 'comb'

    def l_oracle(self, tree):
        (left0, right0, _) = self.stack[-1]
        labels = tree.span_labels(left0, right0)[::-1]
        if len(labels) == 0:
            return 'none'
        else:
            return 'label-' + '-'.join(labels)

    @staticmethod
    def gold_actions(tree):
        n = len(tree.sentence)
        state = Parser(n)
        result = []

        for step in range(2 * n - 1):

            if state.can_combine():
                (left0, right0, _) = state.stack[-1]
                (left1, _, _) = state.stack[-2]
                a, b = tree.enclosing(left0, right0)
                if left1 >= a:
                    result.append('comb')
                    state.combine()
                else:
                    result.append('sh')
                    state.shift()
            else:
                result.append('sh')
                state.shift()

            (left0, right0, _) = state.stack[-1]
            labels = tree.span_labels(left0, right0)[::-1]
            if len(labels) == 0:
                result.append('none')
            else:
                result.append('label-' + '-'.join(labels))
                state.label(labels)

        return result

    @staticmethod
    def training_data(tree):
        """
        Using oracle (for gold sequence), omitting mandatory S-actions
        """
        s_features = []
        l_features = []

        n = len(tree.sentence)
        state = Parser(n)

        for step in range(2 * n - 1):

            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                action = state.s_oracle(tree)
                features = state.s_features()
                s_features.append((features, action))
            state.take_action(action)

            action = state.l_oracle(tree)
            features = state.l_features()
            l_features.append((features, action))
            state.take_action(action)

        return (s_features, l_features)

    @staticmethod
    def exploration(data, fm, network, unk_param=0.75, alpha=1.0, beta=0):
        """
        Only data from this parse, including mandatory S-actions.
            Follow softmax distribution for structural data.
        """
        loss, example = network(data, fm, alpha, beta, unk_param, test=False)
        total_states = len(example["struct_data"]) + len(example["label_data"])
        return loss, total_states, example["accuracy"]

    @staticmethod
    def parse(sentence, fm, network):
        return network._inference(sentence, fm)

    @staticmethod
    def evaluate_corpus(trees, fm, network):
        accuracy = FScore()
        for tree in trees:
            predicted = Parser.parse(tree.sentence, fm, network)
            local_accuracy = predicted.compare(tree)
            accuracy += local_accuracy
        return accuracy

    @staticmethod
    def write_predicted(fname, trees, fm, network):
        """
        Input trees being used only to carry sentences.
        """
        f = open(fname, 'w')
        accuracy = FScore()
        for tree in trees:
            predicted = Parser.parse(tree.sentence, fm, network)
            local_accuracy = predicted.compare(tree)
            accuracy += local_accuracy
            topped = PhraseTree(
                symbol='TOP',
                children=[predicted],
                sentence=predicted.sentence,
            )
            f.write(str(topped))
            f.write('\n')
        f.close()
        return accuracy

    @staticmethod
    def write_raw_predicted(fname, sentences, fm, network):
        f = open(fname, 'w')
        for sentence in sentences:
            predicted = Parser.parse(sentence, fm, network)
            topped = PhraseTree(
                symbol='TOP',
                children=[predicted],
                sentence=predicted.sentence,
            )
            f.write(str(topped))
            f.write('\n')
        f.close()
