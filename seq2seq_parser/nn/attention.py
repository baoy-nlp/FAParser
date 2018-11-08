import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq_parser.utils.ops import batch_slice_select
from seq2seq_parser.utils.ops import sequence_mask


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class BahdanauAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size=None):
        super().__init__()

        self.query_size = query_size
        self.key_size = key_size

        if hidden_size is None:
            hidden_size = key_size

        self.hidden_size = hidden_size

        self.linear_key = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)
        self.linear_logit = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.softmax = BottleSoftmax(dim=1)
        self.tanh = nn.Tanh()

        self._reset_parameters()

    def compute_cache(self, memory):

        return self.linear_key(memory)

    def forward(self, query, memory, cache=None, mask=None):
        """
        :param query: Key tensor.
            with shape [batch_size, input_size]
        :param memory: Memory tensor.
            with shape [batch_size, mem_len, input_size]
        :param mask: Memory mask which the PAD position is marked with true.
            with shape [batch_size, mem_len]
        """

        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True
        else:
            one_step = False

        batch_size, q_len, q_size = query.size()
        _, m_len, m_size = memory.size()

        q = self.linear_query(query.view(-1, q_size))  # [batch_size, q_len, hidden_size]

        if cache is not None:
            k = cache
        else:
            k = self.linear_key(memory.view(-1, m_size))  # [batch_size, m_len, hidden_size]

        # logit = q.unsqueeze(0) + k # [mem_len, batch_size, dim]
        logits = q.view(batch_size, q_len, 1, -1) + k.view(batch_size, 1, m_len, -1)
        logits = self.tanh(logits)
        logits = self.linear_logit(logits.view(-1, self.hidden_size)).view(batch_size, q_len, m_len)

        if mask is not None:
            mask_ = mask.unsqueeze(1)  # [batch_size, 1, m_len]
            logits = logits.masked_fill(mask_, -1e18)

        weights = self.softmax(logits)  # [batch_size, q_len, m_len]

        # [batch_size, q_len, m_len] @ [batch_size, m_len, m_size]
        # ==> [batch_size, q_len, m_size]
        attns = torch.bmm(weights, memory)

        if one_step:
            attns = attns.squeeze(1)  # ==> [batch_size, q_len]

        return attns, weights


class FNNAttention(nn.Module):
    def __init__(self, query_size, key_size=None, hidden_size=None):
        super().__init__()

        self.query_size = query_size
        if key_size is None:
            key_size = query_size

        self.key_size = key_size

        if hidden_size is None:
            hidden_size = key_size

        self.hidden_size = hidden_size

        self.linear_key = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)
        self.linear_logit = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.linear_out = nn.Linear(query_size + key_size, query_size)
        self.softmax = nn.functional.softmax
        self.tanh = nn.Tanh()

    def compute_cache(self, memory):
        return self.linear_key(memory)

    def forward(self, query, memory, cache=None, mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True
        else:
            one_step = False

        batch_size, q_len, q_size = query.size()
        _, m_len, m_size = memory.size()

        q = self.linear_query(query.contiguous().view(-1, q_size))  # [batch_size, q_len, hidden_size]

        if cache is not None:
            k = cache
        else:
            k = self.linear_key(memory.contiguous().view(-1, m_size))  # [batch_size, m_len, hidden_size]

        logits = q.view(batch_size, q_len, 1, -1) + k.view(batch_size, 1, m_len, -1)
        logits = self.tanh(logits)
        logits = self.linear_logit(logits.view(-1, self.hidden_size)).view(batch_size, q_len, m_len)

        if mask is not None:
            mask_ = mask.unsqueeze(1)  # [batch_size, 1, m_len]
            logits = logits.masked_fill(mask_, -1e18)

        weights = self.softmax(logits, dim=-1)  # [batch_size, q_len, m_len]

        # [batch_size, q_len, m_len] @ [batch_size, m_len, m_size]
        # ==> [batch_size, q_len, m_size]
        attns = torch.bmm(weights, memory)

        if one_step:
            attns = attns.squeeze(1)  # ==> [batch_size, q_len]

        combined = torch.cat((query, attns), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, m_size + q_size))).view(batch_size, -1, q_size)

        return output, weights


# class FNNAttention(nn.Module):
#     def __init__(self, dec_dim, enc_dim=None, out_dim=None):
#         super(FNNAttention, self).__init__()
#         self.dec_dim = dec_dim
#         if enc_dim is None:
#             self.enc_dim = dec_dim
#         else:
#             self.enc_dim = enc_dim
#         if out_dim is None:
#             self.out_dim = dec_dim
#         else:
#             self.out_dim = out_dim
#         self.linear_query = nn.Linear(self.dec_dim, 1)
#         self.linear_context = nn.Linear(self.enc_dim, 1)
#         self.linear_out = nn.Linear(self.dec_dim + self.enc_dim, self.out_dim)
#
#     def forward(self, query, key):
#         """
#         :param query: [batch,dec_step,dec_dim]
#         :param key: [batch,enc_step,enc_dim]
#         :return:
#         """
#         batch_size = query.size(0)
#         dec_step = query.size(1)
#         enc_step = key.size(1)
#
#         out_dim = query.size(2) + key.size(2)
#
#         query_score = self.linear_query(query)
#         # [batch,dec_step,1]
#         key_score = self.linear_context(key)
#         # [batch,enc_step,1]
#         expand_q = query_score.expand((batch_size, dec_step, enc_step))
#         expand_k = key_score.transpose(1, 2).expand((batch_size, dec_step, enc_step))
#
#         attn = expand_q + expand_k
#         attn = F.softmax(attn.view(-1, enc_step), dim=1).view(batch_size, -1, enc_step)
#         # target: [batch,dec_step,enc_step]
#
#         mix = torch.bmm(attn, key)
#         # concat -> (batch, out_len, 2*dim)
#         combined = torch.cat((mix, query), dim=2)
#         # output -> (batch, out_len, dim)
#         output = F.tanh(self.linear_out(combined.view(-1, out_dim))).view(batch_size, -1, self.out_dim)
#
#         return output, attn


class DotProductAttention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim):
        super(DotProductAttention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        """
        Args:
            output: decoder state
            context: encoder outputs
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class BothSideAttention(nn.Module):
    def __init__(self, dim):
        super(BothSideAttention, self).__init__()
        reduce = True

        if reduce:
            self.linear = nn.Sequential(
                nn.Linear(dim * 4, dim),
                nn.Tanh()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(dim * 3, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim),
                nn.Tanh()
            )
        self.mask = None

    def forward(self, output, context, split_index):
        """
        Args:
            output: decoder state
            context: encoder outputs
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2)).squeeze(1)
        self.mask = sequence_mask(sequence_length=split_index, max_len=input_size)

        f_attn = attn.data.masked_fill(self.mask.eq(0), -float('inf'))
        f_attn = F.softmax(f_attn.view(-1, input_size), dim=1)
        f_attn[range(batch_size), split_index] = 0.0
        f_attn = f_attn.view(batch_size, -1, input_size)
        a_mix = torch.bmm(f_attn, context)

        self.mask[range(batch_size), split_index] = 0
        b_attn = attn.data.masked_fill(self.mask.eq(1), -float('inf'))
        b_attn = F.softmax(b_attn.view(-1, input_size), dim=1)
        b_attn[range(batch_size), split_index] = 0.0
        b_attn = f_attn.view(batch_size, -1, input_size)
        b_mix = torch.bmm(b_attn, context)

        det_att = batch_slice_select(input=context, dim=1, index=split_index).view(batch_size, 1, -1)

        combined = torch.cat((a_mix, b_mix, det_att, output), dim=2)
        output = self.linear(combined.view(-1, 4 * hidden_size)).view(batch_size, -1, hidden_size)

        return output, attn
