from __future__ import absolute_import
from __future__ import print_function

import argparse
import math
import os
import random
import re
import subprocess
import sys

sys.path.append(".")
import numpy
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from NJUParser.dataset.creater import TreeCreator
from NJUParser.dataset.helper import FScore
from NJUParser.dataset.helper import build_nltktree
from NJUParser.dataset.helper import process_str_tree
from NJUParser.dataset.loader import TreeLoader
from NJUParser.loss.loss_modules import *
from NJUParser.models import build_model
from NJUParser.utils.config_utils import args_to_dict
from NJUParser.utils.config_utils import dict_to_args
from NJUParser.utils.config_utils import yaml_load_dict


def process_args():
    parser = argparse.ArgumentParser(
        description='Syntactic distance based neural parser')
    parser.add_argument("--configpath", type=str, default="/home/user_data/baoy/projects/NJUParser-pytorch/configs/distance.yaml")
    parser.add_argument('--epc', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--pos_embed_dim', type=int, default=400)
    parser.add_argument('--word_embed_dim', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=1200)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.3)
    parser.add_argument('--drope', type=float, default=0.1)
    parser.add_argument('--dropr', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_glove', action='store_true')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--dev_every', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--datapath', type=str, default='../data/ptb')
    parser.add_argument('--savepath', type=str, default='../results')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--model_select', type=str, default='distance_parser')

    parsed_args = parser.parse_args()

    config_dict = yaml_load_dict(parsed_args.configpath)
    parsed_args_dict = args_to_dict(parsed_args)
    if parsed_args_dict['filename'] is not None:
        config_dict['filename'] = parsed_args_dict['filename']

    config_args = dict_to_args(config_dict)
    # set seed and return args
    random.seed(config_args.seed)
    torch.random.manual_seed(config_args.seed)
    if config_args.cuda and torch.cuda.is_available():
        torch.cuda.random.manual_seed(config_args.seed)
    return config_args


def evaluate(model, data, mode='valid'):
    import tempfile
    args = model.args
    arc_dict = model.vocab.arc
    stag_dict = model.vocab.tag
    model.eval()
    if mode == 'valid':
        words, tags, ptags, arcs, dsts = data.batchify(mode, 1)
        _, _, _, _, _, sents, trees = data.valid
    else:
        words, tags, ptags, arcs, dsts = data.batchify(mode, 1)
        _, _, _, _, _, sents, trees = data.test

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_pred_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_ref_path = os.path.join(temp_path.name, "ref_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_pred_path, temp_ref_path))
    temp_pred_file = open(temp_pred_path, "w")
    temp_ref_file = open(temp_ref_path, "w")

    set_loss = 0.0
    set_counter = 0
    set_arc_prec = 0.0
    arc_counter = 0
    set_tag_prec = 0.0
    tag_counter = 0
    for _, (e_words, e_tags, e_stags, e_arcs, e_dsts, e_sents, e_trees) in enumerate(
            zip(words, tags, ptags, arcs, dsts, sents, trees)):

        if args.cuda:
            e_words = e_words.cuda()
            e_tags = e_tags.cuda()
            e_stags = e_stags.cuda()
            e_arcs = e_arcs.cuda()
            e_dsts = e_dsts.cuda()

        mask = (e_words >= 0).float()
        e_words = e_words * mask.long()
        pred_dst, pred_arc, pred_tag = model(e_words, e_stags, mask)

        e_dsts_mask = (e_dsts > 0).float()
        e_loss = rank_loss(pred_dst, e_dsts, e_dsts_mask)
        set_loss += e_loss.item()
        set_counter += 1

        _, pred_arc_idx = torch.max(pred_arc, dim=-1)
        arc_prec = ((e_arcs == pred_arc_idx).float() * e_dsts_mask).sum()
        set_arc_prec += arc_prec.item()
        arc_counter += e_dsts_mask.sum().item()

        _, pred_tag_idx = torch.max(pred_tag, dim=-1)
        pred_tag_idx[0], pred_tag_idx[-1] = -1, -1
        tag_prec = (e_tags == pred_tag_idx).float().sum()
        set_tag_prec += tag_prec.item()
        tag_counter += (e_tags != 0).float().sum().item()

        pred_tree = build_nltktree(
            pred_dst.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_arc_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            pred_tag_idx.data.squeeze().cpu().numpy().tolist()[1:-1],
            e_sents,
            arc_dict.id2word,
            arc_dict.id2word,
            stag_dict.id2word,
            stags=e_stags.data.squeeze().cpu().numpy().tolist()[1:-1]
        )

        temp_pred_file.write(process_str_tree(str(pred_tree)) + '\n')
        temp_ref_file.write(process_str_tree(str(e_trees)) + '\n')

    # execute the evalb command:
    temp_pred_file.close()
    temp_ref_file.close()

    evalb_dir = os.path.join(os.getcwd(), args.eval_path)
    # evalb_dir = args.eval_path
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_ref_path,
        temp_pred_path,
        temp_eval_path)

    subprocess.run(command, shell=True)
    fscore = FScore(math.nan, math.nan, math.nan)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
                break

    temp_path.cleanup()

    set_loss /= set_counter
    set_arc_prec /= arc_counter
    set_tag_prec /= tag_counter

    model.train()

    return set_loss, set_arc_prec, set_tag_prec, fscore.precision, fscore.recall, fscore.fscore


def train():
    args = process_args()

    if args.filename is None:
        # using parameter set the model file name
        filename = sorted(str(args)[10:-1].split(', '))
        filename = [i for i in filename if ('dir' not in i) and
                    ('tblog' not in i) and
                    ('fre' not in i) and
                    ('cuda' not in i) and
                    ('nlookback' not in i)]
        filename = __file__.split('.')[0].split('/')[-1] + '_' + '_'.join(filename) \
            .replace('=', '') \
            .replace('/', '') \
            .replace('\'', '') \
            .replace('..', '') \
            .replace('\"', '')
    else:
        filename = args.filename

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath, exist_ok=True)
    parameter_file = os.path.join(args.savepath, filename + '.th')
    print('model path:', parameter_file)

    print(args)
    print("loading data ...")
    try:
        tree_parsed = TreeLoader(data_path=args.datapath)
        vocab = tree_parsed.vocab
    except:
        TreeCreator(output_path=args.datapath, treebank_path=args.datapath)
        tree_parsed = TreeLoader(data_path=args.datapath)
        vocab = tree_parsed.vocab

    train_log_template = '\repoch {:<3d} batch {:<4d} loss {:<.6f} rank {:<.6f} arc {:<.6f} tag {:<.6f}'
    valid_log_template = '*** epoch {:<3d}  \tloss     \tarc prec \ttag prec \tprecision\trecall   \tlf1      \n' \
                         '{:10}DEV\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\n' \
                         '{:10}TEST\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}\t{:<.6f}'

    # if __name__ == "__main__":

    print("building model...")
    model = build_model(args.model_select, args=args, vocab=vocab)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=args.factor, min_lr=0.000001)

    print(" ")
    num_params = sum([numpy.prod(i.size()) for i in model.parameters()])
    print("Number of params: {0}\n{1:35}{2:35}Size".format(num_params, 'Name', 'Shape'))
    # this includes tied parameters
    print("---------------------------------------------------------------------------")
    for item in model.state_dict().keys():
        this_param = model.state_dict()[item]
        print("{:60}{!s:35}{}".format(item, this_param.size(), numpy.prod(this_param.size())))
    print(" ")

    best_valid_f1 = 0.0
    start_epoch = args.sepc

    print("training ...")

    train_words, train_tags, train_stags, train_arcs, train_distances, train_sents, train_trees = tree_parsed.batchify('train', args.batch_size)
    if args.dev_every == -1:
        args.dev_every = len(train_words)

    for epoch in range(start_epoch, args.epc):
        instance_ids = list(range(len(train_words)))
        random.shuffle(instance_ids)
        epc_train_words = [train_words[i] for i in instance_ids]
        epc_train_tags = [train_tags[i] for i in instance_ids]
        epc_train_stags = [train_stags[i] for i in instance_ids]
        epc_train_arcs = [train_arcs[i] for i in instance_ids]
        epc_train_distances = [train_distances[i] for i in instance_ids]

        for batch_idx, (b_words, b_tags, b_stags, b_arcs, b_dsts) in enumerate(
                zip(
                    epc_train_words,
                    epc_train_tags,
                    epc_train_stags,
                    epc_train_arcs,
                    epc_train_distances,
                )):

            if args.cuda:
                b_words = b_words.cuda()
                b_tags = b_tags.cuda()
                b_stags = b_stags.cuda()
                b_arcs = b_arcs.cuda()
                b_dsts = b_dsts.cuda()

            mask = (b_words >= 0).float()  # unk is zero, pad is -1
            b_words = b_words * mask.long()
            dst_mask = (b_dsts > 0).float()  # just consider the idx which bigger than 0

            optimizer.zero_grad()
            pred_dst, pred_arc, pred_tag = model.forward(b_words, b_stags, mask)
            loss_rank = rank_loss(pred_dst, b_dsts, dst_mask)
            loss_arc = arcloss(pred_arc, b_arcs.view(-1))
            loss_tag = tagloss(pred_tag, b_tags.view(-1))

            loss = loss_rank + loss_arc + loss_tag
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            if (batch_idx + 1) % args.log_every == 0:
                print(train_log_template.format(epoch, batch_idx + 1, loss.item(),
                                                loss_rank.item(), loss_arc.item(),
                                                loss_tag.item()), end=" ")

            if (batch_idx + 1) % args.dev_every == 0:
                print()
                print("Evaluating valid... ")
                valid_loss, valid_arc_prec, valid_tag_prec, valid_precision, valid_recall, valid_f1 = evaluate(model, tree_parsed, 'valid')
                print("Evaluating test... ")
                test_loss, test_arc_prec, test_tag_prec, test_precision, test_recall, test_f1 = evaluate(model, tree_parsed, 'test')
                print(valid_log_template.format(
                    epoch,
                    ' ', valid_loss, valid_arc_prec, valid_tag_prec,
                    valid_precision, valid_recall, valid_f1,
                    ' ', test_loss, test_arc_prec, test_tag_prec,
                    test_precision, test_recall, test_f1))

                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    torch.save({
                        'epoch': epoch,
                        'valid_loss': valid_loss,
                        'valid_precision': valid_precision,
                        'valid_recall': valid_recall,
                        'valid_f1': valid_f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, parameter_file)

                scheduler.step(valid_f1)


if __name__ == "__main__":
    train()
