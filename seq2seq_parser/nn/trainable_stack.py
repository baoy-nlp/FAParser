import torch
import torch.nn as nn

from seq2seq_parser.utils.global_names import GlobalNames


class TrainableStack(nn.Module):
    def __init__(self, push_rnn, pop_rnn):
        super(TrainableStack, self).__init__()
        self.push_rnn = push_rnn
        self.pop_rnn = pop_rnn

    def _transpose_hidden(self, hidden):
        if isinstance(hidden, tuple):
            self.hidden_size = hidden[0].size()[-1]
            hidden = tuple([h.transpose(1, 0).contiguous().view(self.batch_size, 1, -1) for h in hidden])
        else:
            self.hidden_size = hidden.size()[-1]
            hidden = hidden.transpose(1, 0).contiguous().view(self.batch_size, 1, -1)
        return hidden

    def process_hidden(self, push_hidden, pop_hidden, is_pop):
        _push = self._transpose_hidden(push_hidden)
        _pop = self._transpose_hidden(pop_hidden)

        if isinstance(_push, tuple):
            pre_hid = tuple([torch.cat([pu, po], dim=1) for (pu, po) in zip(_push, _pop)])
            hiddens = tuple([h[range(self.batch_size), is_pop, :].view(self.batch_size, -1, self.hidden_size).transpose(1, 0).contiguous() for h in pre_hid])
        else:
            pre_hid = torch.cat([_push, _pop], dim=1)
            hiddens = pre_hid[range(self.batch_size), is_pop, :].view(self.batch_size, -1, self.hidden_size).transpose(1, 0).contiguous()

        return hiddens

    def forward(self, symbol, embedding, hidden):
        self.batch_size = symbol.size()[0]
        is_pop = (symbol.ge(GlobalNames.M) * symbol.lt(GlobalNames.L)).long().squeeze()
        push_o, push_h = self.push_rnn(embedding, hidden)
        pop_o, pop_h = self.pop_rnn(embedding, hidden)

        new_hidden = self.process_hidden(push_h, pop_h, is_pop)
        output = torch.cat([push_o, pop_o], dim=1)
        output = output[range(self.batch_size), is_pop, :].unsqueeze(1)

        return output, new_hidden
