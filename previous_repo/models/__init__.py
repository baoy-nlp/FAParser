"""
parser list
"""
from .fast_parser import DistanceParser
from .faster_parser import TransformerDistanceParser

MODEL_CLS = {
    'transformerdistanceparser': TransformerDistanceParser,
    'distanceparser': DistanceParser
}


def build_model(model: str, **kwargs):
    if model.lower() not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model.lower()](**kwargs)
