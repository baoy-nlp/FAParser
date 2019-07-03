import numpy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from FAParser.dataset.helper import build_nltktree
from FAParser.dataset.helper import process_str_tree
from FAParser.models.parser import Parser
from FAParser.modules.embed_regularize import embedded_dropout
from FAParser.modules.weight_drop import WeightDrop


def run_rnn(inputs, rnn, lengths):
    sorted_idx = numpy.argsort(lengths)[::-1].tolist()
    rnn_input = pack_padded_sequence(inputs[sorted_idx], lengths[sorted_idx], batch_first=True)
    rnn_out, _ = rnn(rnn_input)  # (bsize, ntoken, hidsize*2)
    rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
    rnn_out = rnn_out[numpy.argsort(sorted_idx).tolist()]

    return rnn_out


class DistanceParser(Parser):
    def get_loss(self, example, return_enc_state=False):
        raise NotImplementedError

    def __init__(self, args, vocab, word_embed=None, name="Fast Distance Parser"):
        """
        module include use LSTM encoder, CNN layer,
        :param args:
        :param vocab:
        :param name:
        """
        super().__init__(args, vocab, name)
        dropout = args.drop
        dropoutr = args.dropr
        hidden_dim = args.hidden_dim
        window_size = args.window_size
        word_layer = args.word_layer
        arc_layer = args.arc_layer

        self.hid_size = hidden_dim
        self.arc_size = vocab.arc.size
        self.dropoute = args.drope
        self.drop = nn.Dropout(dropout)
        if word_embed is not None:
            self.word_embed = word_embed
        else:
            self.word_embed = nn.Embedding(vocab.word.size, args.word_embed_dim)
        self.pos_embed = nn.Embedding(vocab.tag.size, args.pos_embed_dim)

        self.word_rnn = nn.LSTM(args.word_embed_dim + args.pos_embed_dim,
                                hidden_dim,
                                num_layers=word_layer,
                                dropout=dropout,
                                bidirectional=args.word_bid,
                                )
        self.word_rnn = WeightDrop(self.word_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.conv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim * 2,
                hidden_dim,
                window_size
            ),
            nn.ReLU()
        )

        self.arc_rnn = nn.LSTM(hidden_dim,
                               hidden_dim,
                               num_layers=arc_layer,
                               dropout=dropout,
                               bidirectional=args.arc_bid,
                               )
        self.arc_rnn = WeightDrop(self.arc_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.terminal_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.non_terminal_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.distance_regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.arc_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab.arc.size),
        )

    def forward(self, words, stags, mask):
        bsz, ntoken = words.size()
        emb_words = embedded_dropout(self.word_embed, words, dropout=self.dropoute if self.training else 0)
        emb_words = self.drop(emb_words)

        emb_stags = embedded_dropout(self.pos_embed, stags, dropout=self.dropoute if self.training else 0)
        emb_stags = self.drop(emb_stags)

        sent_lengths = (mask.sum(dim=1)).data.cpu().numpy().astype('int')
        dst_lengths = sent_lengths - 1
        emb_plus_tag = torch.cat([emb_words, emb_stags], dim=-1)

        word_rnn_output = run_rnn(emb_plus_tag, self.word_rnn, sent_lengths)
        terminal = self.terminal_mapper(word_rnn_output.view(-1, self.hid_size * 2))
        tag_predict = self.arc_classifier(terminal)  # (bsize, ndst, tagsize)
        arc_rnn_input = self.conv1(word_rnn_output.permute(0, 2, 1)).permute(0, 2, 1)  # (bsize, ndst, hidsize)
        arc_rnn_output = run_rnn(arc_rnn_input, self.arc_rnn, dst_lengths)
        non_terminal = self.non_terminal_mapper(arc_rnn_output.view(-1, self.hid_size * 2))
        distance = self.distance_regressor(arc_rnn_output.view(-1, self.hid_size * 2)).squeeze(dim=-1)  # (bsize, ndst)
        arc_predict = self.arc_classifier(non_terminal)  # (bsize, ndst, arcsize)
        return distance.view(bsz, ntoken - 1), arc_predict.contiguous().view(-1, self.arc_size), tag_predict.view(-1, self.arc_size)

    def parse(self, words, stags, sents):
        mask = (words >= 0).float()
        words = words * mask.long()
        pred_dst, pred_arc, pred_tag = self.forward(words, stags, mask)
        _, pred_arc_idx = torch.max(pred_arc, dim=-1)
        _, pred_tag_idx = torch.max(pred_tag, dim=-1)
        pred_tree = build_nltktree(
            pred_dst.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_arc_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_tag_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            sents,
            self.vocab.arc.id2word,
            self.vocab.arc.id2word,
            self.vocab.tag.id2word,
            stags=stags.data.squeeze().cpu().numpy().tolist()[1:-1]
        )
        return process_str_tree(pred_tree)

    def save(self, path):
        super().save(path)
