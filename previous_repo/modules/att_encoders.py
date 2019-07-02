import torch.nn as nn

from .layer_norm import LayerNormalization as LayerNorm
from .sublayers import MultiHeadedAttention, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.layer_norm = LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm.forward(enc_input)
        context, _, _ = self.slf_attn.forward(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn.forward(out)


class AttentionEncoder(nn.Module):
    def __init__(self, n_layers, n_head, model_dim, inner_hid_dim, dropout):
        super(AttentionEncoder, self).__init__()
        self.num_layers = n_layers
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=model_dim, d_inner_hid=inner_hid_dim, n_head=n_head, dropout=dropout)
             for _ in range(n_layers)])

        self.layer_norm = LayerNorm(model_dim)

    def forward(self, inputs, enc_slf_attn_mask):
        out = inputs
        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm.forward(out)

        return out
