from __future__ import print_function, division

import torch

import seq2seq_parser
from seq2seq_parser.data.iterator import BucketIterator
from seq2seq_parser.loss import NLLLoss
from seq2seq_parser.utils.global_names import GlobalNames
from seq2seq_parser.utils.tools import PostProcess
from seq2seq_parser.utils.tree_analysis import eval_s2t_robust, eval_s2b

EVAL_DICT = {
    's2s': eval_s2t_robust,
    's2t': eval_s2t_robust,
    's2b': eval_s2b,
}


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq_parser.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq_parser.tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, seq2seq_parser.src_field_name)
                target_variables = getattr(batch, seq2seq_parser.tgt_field_name)

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy

    def evaluate_tree(self, model, data):
        model.eval()
        pred = []
        ref = []
        src = []

        loss = self.loss
        loss.reset()

        device = None if torch.cuda.is_available() else -1
        batch_iterator = BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        src_vocab = data.fields[seq2seq_parser.src_field_name].vocab
        tgt_vocab = data.fields[seq2seq_parser.tgt_field_name].vocab
        src_pad = src_vocab.stoi[data.fields[seq2seq_parser.src_field_name].pad_token]
        pad = tgt_vocab.stoi[data.fields[seq2seq_parser.tgt_field_name].pad_token]

        pp = PostProcess(
            sos=model.decoder.sos_id,
            eos=model.decoder.eos_id,
            tgt_vocab=tgt_vocab,
            src_vocab=src_vocab,
            src_pad=src_pad,
            tgt_pad=pad,
        )

        def extract_decoder(other, target_variables, input_variables):
            seq = other['sequence']
            result_tensor = torch.cat(seq, dim=1)
            batch_size = input_lengths.size(0)
            for i in range(batch_size):
                predict = result_tensor[i].cpu() if torch.cuda.is_available() else result_tensor[i]
                predict = predict.data.numpy()
                pred.append(pp.extract_single_target(predict[:other['length'][i]]))

                target = target_variables[i].cpu() if torch.cuda.is_available() else target_variables[i]
                target = target.data.numpy()
                ref.append(pp.extract_single_target(target))

                if GlobalNames.use_tag:
                    source = input_variables[0][i].cpu() if torch.cuda.is_available() else input_variables[0][i]
                else:
                    source = input_variables[i].cpu() if torch.cuda.is_available() else input_variables[i]
                source = source.data.numpy()
                src.append(pp.extract_single_source(source))

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, seq2seq_parser.src_field_name)
                target_variables = getattr(batch, seq2seq_parser.tgt_field_name)

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

                # Evaluation
                # for step, step_output in enumerate(decoder_outputs):
                #     target = target_variables[:, step + 1]
                #     loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                extract_decoder(other, target_variables, input_variables)

        accuracy, error = EVAL_DICT[GlobalNames.eval_script](preds=pred, golds=ref)
        print("error count:{}".format(error))

        return loss.get_loss(), accuracy
