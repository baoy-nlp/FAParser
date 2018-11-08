import numpy as np
import torch

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from .features import FeatureMapper


class GlobalNames(object):
    tgt_field_name = 'tgt'
    src_field_name = 'src'
    tag_field_name = 'tag'

    fm_file = "data/vocab.json"
    fm = None
    L = -1
    M = -1
    T = -1
    E = -1
    V = -1
    O = -1

    zero_initial = False

    use_dot_attention = True
    use_fnn_attention = False

    use_tag = False
    use_det = False
    use_biatt = False
    use_parent = False
    use_grammar = False
    use_constrain = False
    use_grammar_rnn = False
    use_grammar_add = False
    use_length = False
    use_ensemble = False
    use_ts = False

    eval_script = 's2t'

    @staticmethod
    def show_fields():
        for name, value in vars(GlobalNames).items():
            if value is not None:
                print('%s=%s' % (name, value))

    @staticmethod
    def get_fm():
        if GlobalNames.fm is None:
            GlobalNames.fm = FeatureMapper.load_json(GlobalNames.fm_file)
        return GlobalNames.fm

    @staticmethod
    def set_controller(opt):
        GlobalNames.fm_file = opt.vocab
        GlobalNames.eval_script = opt.etype
        GlobalNames.use_constrain = opt.constrain
        GlobalNames.use_grammar = opt.grammar
        GlobalNames.use_grammar_add = opt.add
        GlobalNames.use_grammar_rnn = not opt.add
        GlobalNames.use_parent = opt.head
        GlobalNames.use_det = opt.pure_det
        GlobalNames.use_biatt = opt.bi_att
        GlobalNames.use_ensemble = opt.ensemble
        GlobalNames.use_length = opt.use_lc
        GlobalNames.use_ts = opt.ts

        if opt.attention_choice == "fnn":
            GlobalNames.use_fnn_attention = True
            GlobalNames.use_dot_attention = False

    @staticmethod
    def init_global_id(vocab):
        GlobalNames.T = vocab.T
        GlobalNames.M = vocab.M
        GlobalNames.L = vocab.L
        GlobalNames.E = vocab.E
        tag_size = len(vocab) - GlobalNames.L
        GlobalNames.V = tag_size + 1
        GlobalNames.O = len(vocab)


class DecoderMask(object):
    def __init__(self):
        self.output_size = GlobalNames.O
        put_var = 0
        TT = np.array([put_var] * self.output_size)
        TT[:GlobalNames.T] = 1
        TT[GlobalNames.E] = 0
        self.TT = device.Tensor(TT).float().view(1, -1)

        EE = np.array([put_var] * self.output_size)
        EE[GlobalNames.E] = 1
        self.EE = device.Tensor(EE).float().view(1, -1)

        LL = np.array([put_var] * self.output_size)
        LL[GlobalNames.T:GlobalNames.M] = 1
        self.LL = device.Tensor(LL).float().view(1, -1)

        RL = np.array([put_var] * self.output_size)
        RL[GlobalNames.M:GlobalNames.L] = 1
        self.RL = device.Tensor(RL).float().view(1, -1)

        W = np.array([put_var] * self.output_size)
        W[GlobalNames.L:] = 1
        self.W = device.Tensor(W).float().view(1, -1)
