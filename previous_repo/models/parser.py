import os

import torch.nn as nn

from FAParser.modules.nn_utils import *


class Parser(nn.Module):
    """
    Form as the encoder of struct vae, need contain follow functions:
    - beam search for syntax sampling [sampling]
    - score the corresponding examples [score]
    - training loss [forward]
    """

    def __init__(self, args, vocab, name="Parser"):
        """
        :param args:
        :param vocab: contains source word list, target label list
        :param name:
        """
        super(Parser, self).__init__()
        self.args = args
        self.vocab = vocab
        self.module_name = name
        print("get a {}!".format(self.module_name))

    def forward(self, **kwargs):
        """
        used for training
        """
        raise NotImplementedError

    def get_loss(self, example, return_enc_state=False):
        """evaluate the example log score"""
        raise NotImplementedError

    def parse(self, **kwargs):
        """need sample a set of example [src,'tgt'] and it's corresponding score, form as beam decoder"""
        raise NotImplementedError

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
        }

        torch.save(params, path)
