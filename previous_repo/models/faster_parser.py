import numpy
import torch
import torch.nn as nn

from FAParser.dataset.helper import build_nltktree
from FAParser.dataset.helper import process_str_tree
from FAParser.models.parser import Parser
from FAParser.modules.att_encoders import AttentionEncoder
from FAParser.modules.embeddings import Embeddings
from FAParser.modules.tensor_ops import get_long_tensor
from FAParser.modules.tensor_ops import sequence_mask


def encode(inputs: torch.Tensor, encoder: AttentionEncoder, lengths: numpy.array):
    max_length = max(lengths)
    mask = sequence_mask(sequence_length=get_long_tensor(lengths), max_len=max_length)
    batch_size, src_len = mask.size()
    enc_slf_attn_mask = mask.unsqueeze(1).expand(batch_size, src_len, src_len).byte()
    output = encoder.forward(inputs, enc_slf_attn_mask=enc_slf_attn_mask)
    return output


class TransformerDistanceParser(Parser):
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
            self.word_embed = Embeddings(vocab.word.size, args.word_embed_dim, add_position_embedding=True, dropout=self.dropoute)
        self.pos_embed = Embeddings(vocab.tag.size, args.pos_embed_dim, add_position_embedding=False, dropout=self.dropoute)

        self.word_rnn = AttentionEncoder(
            n_layers=word_layer,
            n_head=args.n_head,
            model_dim=self.hid_size,
            inner_hid_dim=args.inner_hid_dim,
            dropout=dropoutr
        )
        # self.word_rnn = WeightDrop(self.word_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.conv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                window_size
            ),
            nn.ReLU()
        )

        self.arc_rnn = AttentionEncoder(
            n_layers=arc_layer,
            n_head=args.n_head,
            model_dim=self.hid_size,
            inner_hid_dim=args.inner_hid_dim,
            dropout=dropoutr
        )
        # self.arc_rnn = WeightDrop(self.arc_rnn, ['weight_hh_l0', 'weight_hh_l1'], dropout=dropoutr)

        self.terminal_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.non_terminal_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.distance_regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
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
        sent_lengths = (mask.sum(dim=1)).data.cpu().numpy().astype('int')
        dst_lengths = sent_lengths - 1
        emb_words = self.word_embed.forward(words)
        emb_stags = self.pos_embed.forward(stags)
        word_input = torch.cat([emb_words, emb_stags], dim=-1)

        word_input = self.drop(word_input)
        word_encode_out = encode(word_input, self.word_rnn, sent_lengths)

        terminal = self.terminal_mapper(word_encode_out.view(-1, self.hid_size))
        tag_predict = self.arc_classifier(terminal)  # (bsize, ndst, tagsize)

        arc_input = self.conv1(word_encode_out.permute(0, 2, 1)).permute(0, 2, 1)  # (bsize, ndst, hidsize)
        arc_encode_out = encode(arc_input, self.arc_rnn, dst_lengths)

        non_terminal = self.non_terminal_mapper(arc_encode_out.view(-1, self.hid_size))
        distance = self.distance_regressor(arc_encode_out.view(-1, self.hid_size)).squeeze(dim=-1)  # (bsize, ndst)
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
