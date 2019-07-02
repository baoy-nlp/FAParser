import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from NJUParser.modules.attention import DotProductionAttention as Attention
from NJUParser.modules.baseRNN import BaseRNN


def reflection(x, dim):
    return x


class RNNDecoder(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 vocab,
                 max_len,
                 input_size,
                 hidden_size,
                 eos_id,
                 sos_id,
                 pad_id=0,
                 n_layers=1,
                 rnn_cell='gru',
                 embed_droprate=0,
                 rnn_droprate=0,
                 use_attention=False,
                 embedding=None,
                 update_embedding=True,
                 use_previous_output=True,
                 ):
        super(RNNDecoder, self).__init__(vocab, max_len, input_size, hidden_size,
                                         embed_droprate, rnn_droprate,
                                         n_layers, rnn_cell)

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=rnn_droprate)

        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id

        if use_attention:
            self.attention = Attention(self.hidden_size)
        self.use_last_output = use_previous_output
        self.soft_input_size = self.hidden_size + self.input_size if use_previous_output else self.hidden_size
        self.out = nn.Linear(self.soft_input_size, self.vocab_size, bias=True)

        if embedding is not None:
            self.embedding = embedding
            self.embedding.weight.requires_grad = update_embedding

    def forward_step(self, input_var, hidden, encoder_outputs, func):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.embed_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention.forward(output, encoder_outputs)
        if self.use_last_output:
            output = torch.cat((output, embedded), dim=-1)
        predicted_softmax = func(self.out(output.contiguous().view(-1, self.soft_input_size)), dim=-1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def decode(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio=1.0):

        decoder_outputs, decoder_hidden, ret_dict, enc_states = self.forward(
            inputs,
            encoder_hidden,
            encoder_outputs,
            function=reflection,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        scores = torch.stack(decoder_outputs)
        return scores

    def score_decoding_results(self, scores, tgt_sents_var):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var:
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        batch_size = scores.size(1)

        if tgt_sents_var.size(0) == batch_size:
            tgt_sents_var = tgt_sents_var.contiguous().transpose(1, 0)

        # (tgt_sent_len * batch_size, tgt_vocab_size)
        log_scores = F.log_softmax(scores.view(-1, scores.size(2)), dim=-1)
        # remove leading <s> in tgt sent, which is not used as the target
        flattened_tgt_sents = tgt_sents_var[1:].contiguous().view(-1)

        # tgt_sent_len * batch_size
        tgt_sent_log_scores = torch.gather(log_scores, 1, flattened_tgt_sents.unsqueeze(1)).squeeze(1)
        tgt_sent_log_scores = tgt_sent_log_scores * (1. - torch.eq(flattened_tgt_sents, self.pad_id).float())  # 0 is pad
        # (batch_size)
        tgt_sent_log_scores = tgt_sent_log_scores.view(-1, batch_size).sum(dim=0)

        return tgt_sent_log_scores

    def score(self, inputs, encoder_hidden, encoder_outputs):
        tgt_token_scores = self.decode(inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio=1.0)
        tgt_sent_log_scores = self.score_decoding_results(tgt_token_scores, inputs)

        return tgt_sent_log_scores

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0.0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[RNNDecoder.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self.valid_args(inputs, encoder_hidden, encoder_outputs,
                                                         function, teacher_forcing_ratio)

        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[RNNDecoder.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     func=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                              func=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[RNNDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[RNNDecoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict, encoder_hidden

    def valid_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        return encoder_hidden
