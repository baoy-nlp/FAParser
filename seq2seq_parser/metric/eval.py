# coding:utf-8
"""
Program: 句法树的打分程序
Description: 比较句法树结构的中的span情况
        TODO:
            //1. 完成最基本的打分功能
            2. 建立配置文件机制
Author: Flyaway - flyaway1217@gmail.com
Date: 2014-07-31 19:07:40
Last modified: 2014-08-30 23:33:13
Python release: 3.3.1
"""
from __future__ import division
from tree import Tree
from tree import Node
from tree import ParseError
from Stack_Queue import Queue


def div(x, y):
    if y == 0:
        return 0.0
    else:
        return x * 1.0 / y * 1.0


# ===============Exceptions======================
class ScoreException(Exception):
    def get_details(self):
        return self.details()


class LengthUnmatch(ScoreException):
    def __init__(self, len_gold_sentence, len_test_sentence):
        self.len_gold_sentence = len_gold_sentence
        self.len_test_sentence = len_test_sentence

    def details(self):
        a = "Length Unmatched !"
        b = "gold sentence:" + str(self.len_gold_sentence)
        c = "test sentence:" + str(self.len_test_sentence)
        s = '\n'.join([a, b, c])
        s += '-' * 30
        return s


class WordsUnmatch(ScoreException):
    def __init__(self, gold_sentence, test_sentence):
        self.gold_sentence = gold_sentence
        self.test_sentence = test_sentence

    def details(self):
        a = "Words Unmatched !"
        b = "gold sentence:" + str(self.gold_sentence)
        c = "test sentence:" + str(self.test_sentence)
        s = '\n'.join([a, b, c])
        s += '-' * 30
        return s


# ================Result Class====================
class Result:
    """The class of result data

    Attributes:

        _staticis: is a dict of statistics:

            ID: the ID of current sentence
            length: the length of the sentence
            state:  the state of the current compare  0:OK,1:skip,2:error
            recall: the recall of the two trees
                    recall = matched bracketing / brackets of gold data
            prec:   the precision of the two trees
                    prec = matched bracketing / brackets of test data
            mathched_brackets: the number of mathched brackets
            gold_brackets: the number of gold brackets
            test_brackets: the number of test brackets
            cross_brackets: the number of cross brackets
            words: the number of unique words
            correct_tags: the number of correct tags
            tag_accracy: the accruacy of tags
    """
    STATISTICS_TABLE = [
        "ID", "length", "state", "recall", "prec", "mathched_brackets", "gold_brackets", "test_brackets",
        "cross_brackets", "words", "correct_tags", "tag_accracy"
    ]
    SUMMARY_TABLE = [
        "Number of sentence", "Number of Error sentence", "Number of Skip  sentence", "Number of Valid sentence", "Bracketing Recall",
        "Bracketing Precision", "Bracketing FMeasure", "Complete match", "Average crossing", "No crossing", "Tagging accuracy"
    ]

    def __init__(self):
        self._staticis = dict()

        # initialize the dict
        for name in Result.STATISTICS_TABLE:
            self._staticis[name] = 0

    def __repr__(self):
        sout = ''
        for name in Result.STATISTICS_TABLE:
            s = name + ":" + str(self._staticis[name]) + '\t'
            sout += s
        return sout

    def __str__(self):
        sout = ''
        for name in Result.STATISTICS_TABLE:
            s = name + ":" + str(self._staticis[name]) + '\t'
            sout += s
        return sout

    def __getattr__(self, name):
        if name == "_staticis":
            return self.__dict__[name]
        elif name in Result.STATISTICS_TABLE:
            return self._staticis.get(name, 0)
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name == "_staticis":
            self.__dict__[name] = value
        elif name in Result.STATISTICS_TABLE:
            self._staticis[name] = value
        else:
            print(name)
            raise AttributeError

    def tostring(self):
        sout = ''
        for name in Result.STATISTICS_TABLE:
            value = self._staticis[name]
            # print(name)
            # print(value)
            if type(value) == int:
                s = '{0: >7d}'.format(value)
            else:
                s = '{0: >9.2f}'.format(value * 100)
            sout += s
        # exit()
        return sout


# ===============================================

class Scorer:
    """The Scorer class.

    This class is a manager of scoring, it can socre tree corpus in a specific configuration.
    Every instance corresponding to a configuration.

    Attributes:
    #TODO
    
    """

    def __init__(self):
        pass

    def _get_span(self, node):
        """ Get the span and plain sentence of the tree

        Args:
            node: the node of the tree
            length: the length of the current span

        Returns:
            A tuple of sent and span:
            (sent,span)

            sent: a list, the partial sentence of the tree
            span: a tuple, the span of the node
        """

        def _help_get_span(node):
            length = length_dict[0]
            if node.isLeaf():
                spans = []
                # spans.append((length,length+1))
                node.span = (length, length + 1)
                length += 1
                length_dict[0] = length
                return ([node.getData()], spans)
            else:
                sent = []
                spans = []
                for child in node.children:
                    sub_sent, sub_spans = _help_get_span(child)
                    sent += sub_sent
                    spans += sub_spans
                node.span = (node.children[0].span[0], node.children[-1].span[-1])
                if node.children[0].isLeaf() == False:
                    spans.append((node.span[0], node.getData(), node.span[-1]))
                return (sent, spans)

        length_dict = {0: 0}
        return _help_get_span(node)

    def _get_tags(self, node):
        """Get the tags of the tree

        Args:
            node: the node the tree

        Returns:
            a list of tags in the order of left to right
        """
        if len(node.children) == 1 and node.children[0].isLeaf() == True:
            return [node.getData()]
        else:
            tags = []
            for child in node.children:
                tags += self._get_tags(child)
            return tags

    def _set_head(self):
        # s = ("   Sent.\t\t\t\t\t\t\t\t\tMatched \t\tBracket\t\tCross\t\t\tCorrect Tag\n"
        #        "ID\t\tLen\t\tStat\tRecal\t\tPrec\tBracket\t\tgold\ttest\tBracket\tWords\tTags\tAccracy\n"
        #     "=======================================================================================================\n")
        s = ("        Sent.                            Matched    Bracket     Cross        Correct Tag\n"
             "     ID     Len   Stat    Recal    Prec  Bracket   gold  test  Bracket  Words  Tags Accracy\n"
             )
        s += "=" * 90
        return s

    def _set_tail(self):
        s = "=" * 90
        s += "\n"
        return s

    def _cal_spans(self, gold_spans, test_spans):
        """Calculate the common span and across span

        Args:
            gold_spans: a list of span in gold tree
            test_spans: a list of span in test tree

        Returns:
            a tuple span_result:
                span_result[0]: the number of common spans
                span_result[1]: the number of crossing spans
        """

        _gold_spans = gold_spans[:]
        unmatched_spans = []
        cross_counter = 0

        # the unmatched spans
        for item in test_spans:
            if item in _gold_spans:
                _gold_spans.remove(item)
            else:
                unmatched_spans.append(item)

        # the crossing spans
        for u in unmatched_spans:
            for g in gold_spans:
                if (u[0] < g[0] and u[-1] > g[0] and u[-1] < g[-1]):
                    cross_counter += 1
                    break
                elif (u[0] > g[0] and u[0] < g[-1] and u[-1] > g[-1]):
                    cross_counter += 1
                    break

        return (len(test_spans) - len(unmatched_spans), cross_counter)

    def score_trees(self, gold_tree, test_tree):
        '''Score the two trees

        Args:
            gold_tree: the gold tree
            test_tres: the test tree

        Returns:
            An instance of Result
        '''
        gold_sentence, gold_spans = self._get_span(gold_tree.root)
        test_sentence, test_spans = self._get_span(test_tree.root)

        gold_tags = self._get_tags(gold_tree.root)
        test_tags = self._get_tags(test_tree.root)

        if len(gold_sentence) != len(test_sentence):
            raise LengthUnmatch(len(gold_sentence), len(test_sentence))

        if gold_sentence != test_sentence:
            raise WordsUnmatch(gold_sentence, test_sentence)

        # statistics
        result = Result()
        common_number, cross_number = self._cal_spans(gold_spans, test_spans)

        correct_tags = 0
        correct_tags = len([gold_tag for gold_tag, test_tag in zip(gold_tags, test_tags) if gold_tag == test_tag])

        result.length = len(gold_sentence)
        result.state = 0
        result.recall = div(common_number, len(gold_spans))
        result.prec = div(common_number, len(test_spans))
        result.mathched_brackets = common_number
        result.gold_brackets = len(gold_spans)
        result.test_brackets = len(test_spans)
        result.cross_brackets = cross_number
        result.words = len(gold_sentence)
        result.correct_tags = correct_tags
        result.tag_accracy = div(correct_tags, len(gold_tags))
        return result

    def score_corpus(self, f_gold, f_test):
        """
        score the treebanks

        Args:
            f_gold: a iterator of gold treebank
            f_test: a iterator of test treebank

        Returns:
            a list of instances of Result
        """
        results = []
        time_parse = 0
        time_cal = 0
        for ID, (gold, test) in enumerate(zip(f_gold, f_test)):
            try:
                gold_tree = Tree(gold)
                test_tree = Tree(test)
                current_result = self.score_trees(gold_tree, test_tree)
            except (WordsUnmatch, LengthUnmatch, ParseError) as e:
                current_result = Result()
                current_result.state = 2
            finally:
                # current_result = Result()
                current_result.ID = ID
                results.append(current_result)
        return results

    def result2string(self, results):
        string = ''
        sout = []
        for item in results:
            sout.append(item.tostring())
        string += '\n'.join(sout)
        return string

    def summary2string(self, summary):
        string = []
        for name, value in zip(Result.SUMMARY_TABLE, summary):
            if type(value) == int:
                string.append(name + ":\t" + '{0:5d}'.format(value))
            else:
                string.append(name + ":\t" + '{0:.4f}'.format(value))
        return '\n'.join(string)

    def summary(self, results):

        """Calculate the summary of resutls

        Args:
            results: a list of result of each sentence

        Returns:
            a list contains all the summary data. The data in the list is ordered by Result.SUMMARY_TABLE.
        """

        summay_list = [0] * len(Result.SUMMARY_TABLE)

        # Number of sentence
        summay_list[0] = len(results)

        # Number of Error sentence
        summay_list[1] = len([item for item in results if item.state == 2])  # 2:Error

        # Number of Skip sentence
        summay_list[2] = len([item for item in results if item.state == 1])  # 1:skip

        correct_results = [item for item in results if item.state == 0]

        # Number of Skip sentence
        sentn = summay_list[0] - summay_list[1] - summay_list[2]
        summay_list[3] = sentn

        # Bracketing Recall: matched brackets / gold brackets
        summay_list[4] = div(sum([item.mathched_brackets for item in correct_results]), sum([item.gold_brackets for item in correct_results]))

        # Bracketing Precision: matched brackets / test brackets
        summay_list[5] = div(sum([item.mathched_brackets for item in correct_results]), sum([item.test_brackets for item in correct_results]))

        # Bracketing FMeasure
        summay_list[6] = div((2 * summay_list[4] * summay_list[5]), (summay_list[4] + summay_list[5]))

        # Complete match
        summay_list[7] = div(len([item for item in correct_results if
                                  item.mathched_brackets == item.gold_brackets and item.mathched_brackets == item.test_brackets]), sentn * 100)

        # Average match
        summay_list[8] = div(sum([item.cross_brackets for item in correct_results]), sentn)

        # No crossing
        summay_list[9] = div(len([item for item in correct_results if item.cross_brackets == 0]), sentn) * 100

        # Tagging accuracy: total correct tags / total words
        summay_list[10] = div(sum([item.correct_tags for item in correct_results]), sum([item.words for item in correct_results]))

        return summay_list

    @staticmethod
    def evalb(gold_file_name, test_file_name):
        self = Scorer()
        gold_file = open(gold_file_name)
        test_file = open(test_file_name)
        results = self.score_corpus(gold_file, test_file)
        summary = self.summary(results)
        strs = []

        strs.append(self._set_head())
        strs.append(self.result2string(results))
        strs.append(self._set_tail())
        strs.append(self.summary2string(summary))
        return '\n'.join(strs)

    @staticmethod
    def evalf(gold_file_name, test_file_name):
        """
        :param gold_file_name:
        :param test_file_name:
        :return:
        """
        self = Scorer()
        gold_file = open(gold_file_name)
        test_file = open(test_file_name)
        results = self.score_corpus(gold_file, test_file)
        summary = self.summary(results)
        return summary[6]


if __name__ == "__main__":
    import sys

    gold_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    out_file_name = sys.argv[3]
    print(Scorer.evalb(gold_file_name, test_file_name))
