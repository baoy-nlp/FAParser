__all__ = [
    "Seq2seq",
    "Transformer",
    "DL4MT",
    "NTransformer",
    "build_model"
]

from .dl4mt import DL4MT
from .n_transformer import NTransformer
from .seq2seq import Seq2seq
from .transformer import Transformer

MODEL_CLS = {
    'NTransformer': NTransformer
}


def build_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)
