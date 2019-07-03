"""
Created by baoy-nlp, 2018-11-09

Contains a series of modules that conduct the seq2seq parsing
"""
from FAParser.models.parser import Parser
from FAParser.modules.nn_utils import *
from FAParser.modules.seq2seq import BaseSeq2seq


class Seq2seqParser(Parser):

    def __init__(self, args, vocab, name="My Base Seq-to-Seq Parser", src_embed=None, tgt_embed=None):
        super().__init__(args, vocab, name)
        self.src_vocab = vocab.src
        self.tgt_vocab = vocab.tgt
        self.seq2seq = BaseSeq2seq(
            args=args,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
        )
        self.decoder = self.seq2seq.decoder
        self.beam_decoder = self.seq2seq.beam_decoder

    def init(self):
        self.seq2seq.init()

    def encode(self, input_var, input_length):
        return self.seq2seq.encode(input_var, input_length)

    def decode(self, input_var, input_length):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def parse(self, src_sent, beam_size=5):
        dmts = len(src_sent) * 3
        src_var = to_input_variable(src_sent, self.src_vocab,
                                    cuda=self.args.cuda, training=False, append_boundary_sym=False, batch_first=True)
        src_length = [len(src_sent)]

        encoder_outputs, encoder_hidden = self.encode(input_var=src_var, input_length=src_length)
        encoder_hidden = self.seq2seq.bridge(encoder_hidden)
        meta_data = self.beam_decoder.beam_search(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            beam_size=beam_size,
            decode_max_time_step=dmts
        )
        topk_sequence = meta_data['sequence']
        topk_score = meta_data['score'].squeeze()

        completed_hypotheses = torch.cat(topk_sequence, dim=-1)

        number_return = completed_hypotheses.size(0)
        final_result = []
        final_scores = []
        for i in range(number_return):
            hyp = completed_hypotheses[i, :].data.tolist()
            res = id2word(hyp, self.tgt_vocab)
            if len(res) > 0:
                final_result.append(res)
                final_scores.append(topk_score[i].item())
        return final_result, final_scores

    def batch_greedy_parse(self, examples, to_word=True):
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
        else:
            src_words = examples.src

        src_var = to_input_variable(src_words, self.src_vocab, cuda=args.cuda, batch_first=True)
        src_length = [len(c) for c in src_words]

        encoder_outputs, encoder_hidden = self.encode(input_var=src_var, input_length=src_length)
        encoder_hidden = self.seq2seq.bridge(encoder_hidden)
        decoder_output, decoder_hidden, ret_dict, _ = self.decoder.forward(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            teacher_forcing_ratio=0.0
        )

        result = torch.stack(ret_dict['sequence']).squeeze()
        final_result = []
        example_nums = result.size(1)
        if to_word:
            for i in range(example_nums):
                hyp = result[:, i].data.tolist()
                res = id2word(hyp, self.tgt_vocab)
                seems = [[res], [len(res)]]
                final_result.append(seems)
        return final_result

    def batch_beam_parse(self, examples, beam_size=5, to_word=True):
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
        else:
            src_words = examples.src

        src_var = to_input_variable(src_words, self.src_vocab, cuda=args.cuda, batch_first=True)
        src_length = [len(c) for c in src_words]
        dmts = len(src_words[0]) * 3

        encoder_outputs, encoder_hidden = self.encode(input_var=src_var, input_length=src_length)
        encoder_hidden = self.seq2seq.bridge(encoder_hidden)
        meta_data = self.beam_decoder.beam_search(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            beam_size=beam_size,
            decode_max_time_step=dmts
        )

    def get_loss(self, examples, return_enc_state=False):
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
            tgt_words = [e.tgt for e in examples]
        else:
            src_words = examples.src
            tgt_words = examples.tgt

        src_var = to_input_variable(src_words, self.src_vocab, cuda=args.cuda, batch_first=True)
        tgt_var = to_input_variable(tgt_words, self.tgt_vocab, cuda=args.cuda, append_boundary_sym=True, batch_first=True)
        src_length = [len(c) for c in src_words]

        encoder_outputs, encoder_hidden = self.encode(input_var=src_var, input_length=src_length)
        encoder_hidden = self.seq2seq.bridge(encoder_hidden)
        scores = self.decoder.get_loss(inputs=tgt_var, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs)
        enc_states = self.decoder.init_state(encoder_hidden)

        h = torch.cat([enc_states[i] for i in range(enc_states.size(0))], dim=-1)

        if return_enc_state:
            return scores, h
        else:
            return scores

    def save(self, path):
        super().save(path)
