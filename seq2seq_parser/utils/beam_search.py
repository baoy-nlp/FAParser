import torch
import numpy as np
from seq2seq_parser.data.vocabulary import EOS, PAD
from .common_utils import GlobalNames

__all__ = [
    'tile_batch',
    'mask_scores',
    'tensor_gather_helper',
    'reranking_beams'
]

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


def tile_batch(x, multiplier, batch_dim=0):
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim + 1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled


def mask_scores(scores, beam_mask):
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[EOS] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + \
             torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def tensor_gather_helper(gather_indices,
                         gather_from,
                         batch_size,
                         beam_size,
                         gather_shape):
    range_ = (torch.arange(0, batch_size) * beam_size).long()

    if GlobalNames.USE_GPU:
        range_ = range_.cuda()

    gather_indices_ = (gather_indices + torch.unsqueeze(range_, 1)).view(-1)

    output = torch.index_select(gather_from.view(*gather_shape), 0, gather_indices_)

    out_size = gather_from.size()[:1 + len(gather_shape)]

    return output.view(*out_size)


def reranking_beams(word_ids, scores):
    word_ids = word_ids.cpu().numpy()
    scores = scores.cpu().numpy()

    # Reranking beams
    reranked_beams = np.argsort(scores, axis=1)
    reranked_word_ids = np.ones_like(word_ids) * PAD

    for b in range(scores.shape[0]):
        for ii in reranked_beams[b]:
            reranked_word_ids[b, ii] = word_ids[b, ii]

    reranked_word_ids = reranked_word_ids.tolist()

    return reranked_word_ids
