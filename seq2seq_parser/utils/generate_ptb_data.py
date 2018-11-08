import os
import sys
sys.path.append(".")
from seq2seq_parser.utils.phrase_tree import PhraseTree
from .tools import write_docs
from .tree_linearization import tree_to_s2t


def generate_training_data(tree_files):
    data = []
    with open(tree_files, 'r') as tree_strs:
        for line in tree_strs:
            word, token = tree_to_s2t(line)
            t = PhraseTree.parse(line)
            tag = " ".join([item[1] for item in t.sentence])
            data.append(
                "\t".join([word, tag, token])
            )
        return data


def generate_from_ptb(ptb_root, tgt_root):
    train_data = generate_training_data(ptb_root + "/train.clean")
    write_docs(docs=train_data, fname=tgt_root + "/train.s2t.new")
    dev_data = generate_training_data(ptb_root + "/dev.clean")
    write_docs(docs=dev_data, fname=tgt_root + "/dev.s2t.new")
    test_data = generate_training_data(ptb_root + "/test.clean")
    write_docs(docs=test_data, fname=tgt_root + "/test.s2t.new")


def generate_for_s2s(tree_files):
    src = []
    tgt = []
    with open(tree_files, 'r') as tree_strs:
        for line in tree_strs:
            items = line.strip().split("\t")
            src.append(items[0])
            tgt.append(items[-1])

    write_docs(docs=src, fname=tree_files + ".src")
    write_docs(docs=tgt, fname=tree_files + ".tgt")


if __name__ == "__main__":
    # generate_from_ptb(ptb_root="../../data/ptb", tgt_root="../../data/s2t.new")
    root_path = sys.argv[1]
    for root, _, files in os.walk(root_path):
        for file in files:
            generate_for_s2s(root + "/" + file)
