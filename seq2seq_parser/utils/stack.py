import numpy as np
import torch

from .global_names import GlobalNames
from .ops import batch_elements_select
from .ops import batch_slice_select
from .ops import batch_slice_set
from .ops import get_tensor


class TensorStack(object):
    def __init__(self):
        self.has_init = False
        self.step = 0

    def push(self, *arg):
        raise NotImplementedError


class ConstrainStack(TensorStack):
    REPEAT_TOLERANCE = 5

    def __init__(self):
        super(ConstrainStack, self).__init__()
        self.length_info = None

    def push(self, symbol: torch.Tensor):
        """
        only input is reduce/label, update the topper

        _new_par = is_reduce*time_parent + is_label*time_pointer + (is_tag+is_

        """
        if self.length_info is not None:
            self.length_info.push(symbol)
        if not self.has_init:
            self._init_mask(GlobalNames.O)
            self._init_pointer(symbol)
            self.cond = self._cond(symbol)
        else:
            self.step += 1
            self.cond = self._cond(symbol)

            is_pad, is_tag, is_reduce, is_label, is_repeat = self.cond
            is_other = (is_tag + is_pad)

            _top = self.pointer_top
            _parent = self.pointer_par
            _time = get_tensor([self.step] * self.batch_size).long().view(-1, 1)
            _grands = batch_elements_select(input=self.parent_history, index=_parent).view(-1, 1)

            new_parent = is_other * _parent + is_label * _top + is_reduce * _grands
            new_topper = is_other * _top + is_label * _time + is_reduce * _parent

            self.label_repeat = self.label_repeat * is_repeat + is_repeat
            self._update_pointer(_new_parent=new_parent, _new_topper=new_topper, symbol=symbol)

    @property
    def mask(self):
        """
        Return the mask of next time symbols requires
        """
        has_label = self.pointer_par.ge(0).float()
        prev_pad = self.cond[0].float()
        prev_reduce = self.cond[2].float()
        prev_label = self.cond[3].float()

        tag_mask = has_label * self.length_info.length_mask() if self.length_info is not None else has_label
        end_mask = (1 - has_label) * prev_reduce
        reduce_mask = has_label * (1 - prev_label)

        _mask = tag_mask * self.could_tag + \
                prev_pad * self.could_pad + \
                end_mask * self.could_end + \
                reduce_mask * self.could_reduce

        could_repeat = self.label_repeat.lt(ParentStack.REPEAT_TOLERANCE).float()

        if self.step == 0:
            _mask += prev_pad * self.could_label
        else:
            _mask += (1 - prev_pad) * could_repeat * self.could_label

        return _mask

    def _cond(self, symbol: torch.Tensor):
        is_pad = symbol.lt(GlobalNames.T)
        is_tag = symbol.ge(GlobalNames.L)
        is_reduce = symbol.ge(GlobalNames.M) * symbol.lt(GlobalNames.L)
        is_label = symbol.ge(GlobalNames.T) * symbol.lt(GlobalNames.M)
        is_repeat = is_label * (symbol.eq(self.prev_symbol))
        return is_pad.long(), is_tag.long(), is_reduce.long(), is_label.long(), is_repeat.long()

    def _init_mask(self, output_size):
        mask_var = 0
        act_var = 1
        initial = np.array([mask_var] * output_size)

        TT = initial.copy()
        TT[:GlobalNames.T] = act_var
        TT[GlobalNames.E] = mask_var
        self.could_pad = get_tensor(TT).float().view(1, -1)

        EE = initial.copy()
        EE[GlobalNames.E] = act_var
        self.could_end = get_tensor(EE).float().view(1, -1)

        LL = initial.copy()
        LL[GlobalNames.T:GlobalNames.M] = act_var
        self.could_label = get_tensor(LL).float().view(1, -1)

        RL = initial.copy()
        RL[GlobalNames.M:GlobalNames.L] = act_var
        self.could_reduce = get_tensor(RL).float().view(1, -1)

        W = initial.copy()
        W[GlobalNames.L:] = act_var
        self.could_tag = get_tensor(W).float().view(1, -1)

    def _init_pointer(self, symbol: torch.Tensor):
        self.batch_size = symbol.size()[0]

        self.prev_symbol = symbol
        self.pointer_top = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.pointer_par = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.parent_history = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.label_repeat = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.has_init = True

    def _update_pointer(self, _new_parent, _new_topper, symbol: torch.Tensor):
        self.parent_history = torch.cat([self.parent_history, _new_parent], dim=-1)
        self.pointer_top = _new_topper
        self.pointer_par = _new_parent
        self.prev_symbol = symbol


class ParentStack(TensorStack):
    REPEAT_TOLERANCE = 5

    def __init__(self):
        super(ParentStack, self).__init__()
        self.length_info = None

    def push(self, symbol: torch.Tensor):
        """
        only input is reduce/label, update the topper

        _new_par = is_reduce*time_parent + is_label*time_pointer + (is_tag+is_

        """
        if self.length_info is not None:
            self.length_info.push(symbol)
        if not self.has_init:
            self._init_mask(GlobalNames.O)
            self._init_pointer(symbol)
            self.cond = self._cond(symbol)
        else:
            self.step += 1
            self.cond = self._cond(symbol)

            is_pad, is_tag, is_reduce, is_label, is_repeat = self.cond
            is_other = (is_tag + is_pad)

            _top = self.pointer_top
            _parent = self.pointer_par
            _time = get_tensor([self.step] * self.batch_size).long().view(-1, 1)
            _grands = batch_elements_select(input=self.parent_history, index=_parent).view(-1, 1)

            new_parent = is_other * _parent + is_label * _top + is_reduce * _grands
            new_topper = is_other * _top + is_label * _time + is_reduce * _parent

            self.label_repeat = self.label_repeat * is_repeat + is_repeat
            self._update_pointer(_new_parent=new_parent, _new_topper=new_topper, symbol=symbol)

    @property
    def mask(self):
        """
        Return the mask of next time symbols requires
        """
        has_label = self.pointer_par.ge(0).float()
        prev_pad = self.cond[0].float()
        prev_reduce = self.cond[2].float()
        prev_label = self.cond[3].float()

        tag_mask = has_label * self.length_info.length_mask() if self.length_info is not None else has_label
        end_mask = (1 - has_label) * prev_reduce
        reduce_mask = has_label * (1 - prev_label)

        _mask = tag_mask * self.could_tag + \
                prev_pad * self.could_pad + \
                end_mask * self.could_end + \
                reduce_mask * self.could_reduce

        could_repeat = self.label_repeat.lt(ParentStack.REPEAT_TOLERANCE).float()

        if self.step == 0:
            _mask += prev_pad * self.could_label
        else:
            _mask += (1 - prev_pad) * could_repeat * self.could_label

        return _mask

    @property
    def top_symbol(self):
        symbol = batch_elements_select(input=self.symbol_history, index=self.pointer_top).view(-1, 1)
        return symbol

    def _cond(self, symbol: torch.Tensor):
        is_pad = symbol.lt(GlobalNames.T)
        is_tag = symbol.ge(GlobalNames.L)
        is_reduce = symbol.ge(GlobalNames.M) * symbol.lt(GlobalNames.L)
        is_label = symbol.ge(GlobalNames.T) * symbol.lt(GlobalNames.M)
        is_repeat = is_label * (symbol.eq(self.prev_symbol))
        return is_pad.long(), is_tag.long(), is_reduce.long(), is_label.long(), is_repeat.long()

    def _init_mask(self, output_size):
        mask_var = 0
        act_var = 1
        initial = np.array([mask_var] * output_size)

        TT = initial.copy()
        TT[:GlobalNames.T] = act_var
        TT[GlobalNames.E] = mask_var
        self.could_pad = get_tensor(TT).float().view(1, -1)

        EE = initial.copy()
        EE[GlobalNames.E] = act_var
        self.could_end = get_tensor(EE).float().view(1, -1)

        LL = initial.copy()
        LL[GlobalNames.T:GlobalNames.M] = act_var
        self.could_label = get_tensor(LL).float().view(1, -1)

        RL = initial.copy()
        RL[GlobalNames.M:GlobalNames.L] = act_var
        self.could_reduce = get_tensor(RL).float().view(1, -1)

        W = initial.copy()
        W[GlobalNames.L:] = act_var
        self.could_tag = get_tensor(W).float().view(1, -1)

    def _init_pointer(self, symbol: torch.Tensor):
        self.batch_size = symbol.size()[0]

        self.symbol_history = symbol
        self.prev_symbol = symbol
        self.pointer_top = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.pointer_par = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.parent_history = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.label_repeat = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.has_init = True

    def _update_pointer(self, _new_parent, _new_topper, symbol: torch.Tensor):

        self.symbol_history = torch.cat([self.symbol_history, symbol], dim=-1)
        self.parent_history = torch.cat([self.parent_history, _new_parent], dim=-1)
        self.pointer_top = _new_topper
        self.pointer_par = _new_parent
        self.prev_symbol = symbol


class GrammarStack(TensorStack):
    def __init__(self):
        super(GrammarStack, self).__init__()
        self.length_info = None

    def push(self, symbol: torch.Tensor, parent: torch.Tensor, cur: torch.Tensor):
        """
        Args:
            symbols: [batch,1]
            hidden: [1,batch,hidden_size]
        """
        if self.length_info is not None:
            self.length_info.push(symbol)
        self.batch_size = symbol.size()[0]
        symbol = symbol.contiguous().view(-1, 1)
        parent = parent.contiguous().view(self.batch_size, 1, -1)
        cur = cur.contiguous().view(self.batch_size, 1, -1)
        if not self.has_init:
            self.init(symbol, cur)
            self.cond = self._cond(symbol)
        else:
            self.step += 1
            self.cond = self._cond(symbol)

            is_pad, is_tag, is_reduce, is_label, is_repeat = self.cond
            is_other = (is_tag + is_pad)
            _top = self.pointer_top
            _parent = self.pointer_par
            _time = get_tensor([self.step] * self.batch_size).long().view(-1, 1)
            _grands = batch_elements_select(input=self.parent_history, index=_parent).view(-1, 1)

            new_parent = is_other * _parent + is_label * _top + is_reduce * _grands
            new_topper = is_other * _top + is_label * _time + is_reduce * _parent

            self.label_repeat = self.label_repeat * is_repeat + is_repeat
            self.update(_new_parent=new_parent, _new_topper=new_topper, symbol=symbol, parent=parent, cur=cur)

    def init(self, symbol, hidden):
        self._init_mask(GlobalNames.O)
        self._init_pointer(symbol)
        self.hidden_history = hidden

    def update(self, _new_parent, _new_topper, symbol: torch.Tensor, parent, cur):
        self.hidden_history = torch.cat([self.hidden_history, cur], dim=1)
        self.hidden_history = batch_slice_set(input=self.hidden_history, dim=1, index=self.pointer_top, to_set=parent.squeeze())

        self.symbol_history = torch.cat([self.symbol_history, symbol], dim=-1)
        self.parent_history = torch.cat([self.parent_history, _new_parent], dim=-1)
        self.pointer_top = _new_topper
        self.pointer_par = _new_parent
        self.prev_symbol = symbol

    def pop(self):
        """
        :return hidden with [layer,batch,hidden_size]
        """
        hidden = batch_slice_select(input=self.hidden_history, dim=1, index=self.pointer_top).view(1, self.batch_size, -1)
        return hidden

    def top_symbol(self):
        symbol = batch_elements_select(input=self.symbol_history, index=self.pointer_top).view(-1, 1)
        return symbol

    @property
    def mask(self):
        """
        Return the mask of next time symbols requires
        """
        has_label = self.pointer_par.ge(0).float()
        prev_pad = self.cond[0].float()
        prev_reduce = self.cond[2].float()
        prev_label = self.cond[3].float()

        tag_mask = has_label * self.length_info.length_mask() if self.length_info is not None else has_label
        end_mask = (1 - has_label) * prev_reduce
        reduce_mask = has_label * (1 - prev_label)

        _mask = tag_mask * self.could_tag + \
                prev_pad * self.could_pad + \
                end_mask * self.could_end + \
                reduce_mask * self.could_reduce

        could_repeat = self.label_repeat.lt(ParentStack.REPEAT_TOLERANCE).float()

        if self.step == 0:
            _mask += prev_pad * self.could_label
        else:
            _mask += (1 - prev_pad) * could_repeat * self.could_label

        return _mask

    def _cond(self, symbol: torch.Tensor):
        is_pad = symbol.lt(GlobalNames.T)
        is_tag = symbol.ge(GlobalNames.L)
        is_reduce = symbol.ge(GlobalNames.M) * symbol.lt(GlobalNames.L)
        is_label = symbol.ge(GlobalNames.T) * symbol.lt(GlobalNames.M)
        is_repeat = is_label * (symbol.eq(self.prev_symbol))
        return is_pad.long(), is_tag.long(), is_reduce.long(), is_label.long(), is_repeat.long()

    def _init_mask(self, output_size):
        mask_var = 0
        act_var = 1
        initial = np.array([mask_var] * output_size)

        TT = initial.copy()
        TT[:GlobalNames.T] = act_var
        TT[GlobalNames.E] = mask_var
        self.could_pad = get_tensor(TT).float().view(1, -1)

        EE = initial.copy()
        EE[GlobalNames.E] = act_var
        self.could_end = get_tensor(EE).float().view(1, -1)

        LL = initial.copy()
        LL[GlobalNames.T:GlobalNames.M] = act_var
        self.could_label = get_tensor(LL).float().view(1, -1)

        RL = initial.copy()
        RL[GlobalNames.M:GlobalNames.L] = act_var
        self.could_reduce = get_tensor(RL).float().view(1, -1)

        W = initial.copy()
        W[GlobalNames.L:] = act_var
        self.could_tag = get_tensor(W).float().view(1, -1)

    def _init_pointer(self, symbol: torch.Tensor):
        self.symbol_history = symbol
        self.prev_symbol = symbol
        self.pointer_top = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.pointer_par = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.parent_history = get_tensor([-1] * self.batch_size).long().view(-1, 1)
        self.label_repeat = get_tensor([0] * self.batch_size).long().view(-1, 1)
        self.has_init = True


class LengthInfo(object):
    def __init__(self, max_length):
        self.batch_size = len(max_length)
        self.max_length = get_tensor(max_length).long().view(-1, 1)
        self.cur_length = get_tensor([0] * self.batch_size).long().view(-1, 1)

    def push(self, symbol: torch.Tensor):
        symbol = symbol.view(-1, 1)
        is_word = symbol.ge(GlobalNames.L).long()
        in_num = self.cur_length.lt(self.max_length - 1).long()
        self.cur_length += is_word * in_num

    def length_mask(self):
        return self.cur_length.lt(self.max_length).float()

    def cur_site(self):
        return self.cur_length

    def extract_context(self, encoder_outputs):
        return batch_slice_select(input=encoder_outputs, dim=1, index=self.cur_length).view(self.batch_size, 1, -1)
