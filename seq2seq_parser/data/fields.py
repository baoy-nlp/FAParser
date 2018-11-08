import logging

import torchtext
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from .dataset import Dataset
from .vocab import TargetVocab

from seq2seq_parser.utils.global_names import GlobalNames


class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """

        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preproces sing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        sources = []
        fm = GlobalNames.get_fm()
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        label = defaultdict(int)
        right = defaultdict(int)
        tag = defaultdict(int)
        for data in sources:
            for x in data:
                for tok in x:
                    if fm.is_tag(tok):
                        tag[tok] = 1
                    elif tok.startswith("/"):
                        right[tok] = 1
                    else:
                        label[tok] = 1

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, TargetField.SYM_SOS,
                            TargetField.SYM_EOS]
            if tok is not None))

        for tok in [self.unk_token, self.pad_token, TargetField.SYM_SOS,
                    TargetField.SYM_EOS]:
            if tok in label:
                label.pop(tok)

        self.vocab = TargetVocab(label, right, tag, specials=specials, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
        self.vocab.E = self.eos_id
        GlobalNames.init_global_id(self.vocab)
