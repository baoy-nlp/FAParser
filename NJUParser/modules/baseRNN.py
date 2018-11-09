""" A base class for RNN. """
import torch.nn as nn


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, embed_size, hidden_size, embed_drop, layer_drop, rnn_layer, rnn_type):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = rnn_layer
        self.embed_dropout = nn.Dropout(p=embed_drop)
        self.rnn_droprate = layer_drop if rnn_layer > 1 else 0.0
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        if rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif rnn_type.lower() == 'rnn':
            self.rnn_cell = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_type))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
