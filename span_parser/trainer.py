from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optimize

from span_parser.global_names import GlobalNames
from span_parser.measures import FScore
from span_parser.model import Network
from span_parser.parser import Parser
from span_parser.phrase_tree import PhraseTree
from span_parser.vocab import Vocab


def generate_vocab(args):
    if args.vocab is not None:
        fm = Vocab.load_json(args.vocab)
    elif args.train is not None:
        fm = Vocab(args.train)
        if args.vocab_output is not None:
            fm.save_json(args.vocab_output)
            print('Wrote vocabulary file {}'.format(args.vocab_output))
            sys.exit()
    else:
        print('Must specify either --vocab-file or --train-data.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()
    return fm


def test(fm, args):
    test_trees = PhraseTree.load_trees(args.test)
    print('Loaded test trees from {}'.format(args.test))
    network = torch.load(args.model)
    print('Loaded model from: {}'.format(args.model))
    accuracy = Parser.evaluate_corpus(test_trees, fm, network)
    print('Accuracy: {}'.format(accuracy))


def parse_sentence(fm, args, sents):
    network = torch.load(args.model)
    print('Loaded model from: {}'.format(args.model))
    predicted = Parser.parse(sents, fm, network)
    predicted.draw_tree('pred.png')


def rerank(fm, args, network=None):
    if network is None:
        network = torch.load(args.model)
        print('Loaded model from: {}'.format(args.model))

    gold = PhraseTree.load_trees(args.gold)
    kbest = PhraseTree.load_kbests(args.kbest, fm)
    res = []
    print('reranking')
    for onebest, g in zip(kbest, gold):
        # gold_score = network.force_decoding(fm.gold_data(g), fm)
        # print(gold_score)
        scores = np.zeros(len(onebest), dtype='float32')
        for i, data in enumerate(onebest):
            scores[i] = force_decoding(network, data, fm)
        maxid = np.argmax(scores)
        # print(scores)
        # print(maxid)
        res.append(onebest[maxid])

    accuracy = FScore()
    baseline = FScore()
    for p, g in zip(kbest, gold):
        local_accuracy = p[0]['tree'].compare(g)
        baseline += local_accuracy

    for p, g in zip(res, gold):
        local_accuracy = p['tree'].compare(g)
        accuracy += local_accuracy
    print(accuracy)
    print(baseline)

    return accuracy


def force_decoding(model, data, fm):
    tree = data['tree']
    sentence = tree.sentence
    n = len(sentence)
    word_ids, tag_ids = fm.index_sentences(sentence)
    fwd, back = model.infer(word_ids, tag_ids, test=True)
    state = Parser(n)
    total_score = 0.
    for step in range(2 * n - 1):
        if not state.can_combine():
            action = "sh"
        elif not state.can_shift():
            action = "comb"
        else:
            left, right = state.s_features()
            scores = model.evaluate_action(
                fwd,
                back,
                left,
                right,
                'struct',
                test=True,
            )
            probs = F.softmax(scores, dim=0)
            action_idx = fm.s_action_index(state.s_oracle(tree))
            if GlobalNames.use_gpu:
                probs = probs.cpu().data.numpy()
            else:
                probs = probs.data.numpy()
            total_score += np.log(probs[action_idx])
            action = state.s_oracle(tree)
        state.take_action(action)

        left, right = state.l_features()
        scores = model.evaluate_action(
            fwd,
            back,
            left,
            right,
            'label',
            test=True,
        )
        probs = F.softmax(scores, dim=0)
        action_idx = fm.l_action_index(state.l_oracle(tree))

        if GlobalNames.use_gpu:
            probs = probs.cpu().data.numpy()
        else:
            probs = probs.data.numpy()

        total_score += np.log(probs[action_idx])
        action = state.l_oracle(tree)
        state.take_action(action)

    return total_score


def train(fm, args):
    train_data_file = args.train
    dev_data_file = args.dev
    epochs = args.epochs
    batch_size = args.batch_size
    unk_param = args.unk_param
    alpha = args.alpha
    beta = args.beta
    model_save_file = args.model

    print("this is train mode")
    start_time = time.time()

    network = Network(fm, args)

    optimizer = optimize.Adadelta(network.parameters(), eps=1e-7, rho=0.99)
    if GlobalNames.use_gpu:
        network.cuda()

    training_data = fm.gold_data_from_file(train_data_file)
    num_batches = -(-len(training_data) // batch_size)
    print('Loaded {} training sentences ({} batches of size {})!'.format(
        len(training_data),
        num_batches,
        batch_size,
    ))
    parse_every = -(-num_batches // 4)

    dev_trees = PhraseTree.load_trees(dev_data_file)
    print('Loaded {} validation trees!'.format(len(dev_trees)))

    best_acc = FScore()

    for epoch in range(1, epochs + 1):
        print('........... epoch {} ...........'.format(epoch))

        total_cost = 0.0
        total_states = 0
        training_acc = FScore()

        np.random.shuffle(training_data)

        for b in range(num_batches):
            network.zero_grad()
            batch = training_data[(b * batch_size): ((b + 1) * batch_size)]
            batch_loss = None
            for example in batch:
                example_Loss, example_states, acc = Parser.exploration(example, fm, network, alpha=alpha, beta=beta, unk_param=unk_param)
                total_states += example_states
                if batch_loss is not None:
                    batch_loss += example_Loss
                else:
                    batch_loss = example_Loss
                training_acc += acc
            if GlobalNames.use_gpu:
                total_cost += batch_loss.cpu().data.numpy()[0]
            else:
                total_cost += batch_loss.data.numpy()[0]
            batch_loss.backward()
            optimizer.step()

            mean_cost = total_cost / total_states

            print(
                '\rBatch {}  Mean Cost {:.4f} [Train: {}]'.format(
                    b,
                    mean_cost,
                    training_acc,
                ),
                end='',
            )
            sys.stdout.flush()

            if ((b + 1) % parse_every) == 0 or b == (num_batches - 1):
                dev_acc = Parser.evaluate_corpus(
                    dev_trees,
                    fm,
                    network,
                )
                print(' [Dev: {}]'.format(dev_acc))

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    s = round(dev_acc.fscore(), 2)
                    temp_save_file = model_save_file.replace('.model', '{}.model'.format(s))
                    torch.save(network, temp_save_file)
                    print(' [saved model: {}]'.format(temp_save_file))
                    # rerank(fm,args)
        current_time = time.time()
        runmins = (current_time - start_time) / 60.
        print('  Elapsed time: {:.2f}m'.format(runmins))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Span-Based Constituency Parser')
    parser.add_argument(
        '--gpu-ids',
        dest='gpu_ids',
        default='-1',
        type=str,
        help="-1",
    )
    parser.add_argument(
        '--model',
        dest='model',
        help='File to save or load model.',
    )
    parser.add_argument(
        '--train',
        dest='train',
        help='Training trees. PTB (parenthetical) format.',
    )
    parser.add_argument(
        '--test',
        dest='test',
        help=(
            'Evaluation trees. PTB (parenthetical) format.'
            ' Omit for training.'
        ),
    )
    parser.add_argument(
        '--dev',
        dest='dev',
        help=(
            'Validation trees. PTB (parenthetical) format.'
            ' Required for training'
        ),
    )
    parser.add_argument(
        '--vocab',
        dest='vocab',
        help='JSON file from which to load vocabulary.',
    )
    parser.add_argument(
        '--write-vocab',
        dest='vocab_output',
        help='Destination to save vocabulary from training data.',
    )
    parser.add_argument(
        '--word-dims',
        dest='word_dims',
        type=int,
        default=50,
        help='Embedding dimesions for word forms. (DEFAULT=50)',
    )
    parser.add_argument(
        '--tag-dims',
        dest='tag_dims',
        type=int,
        default=20,
        help='Embedding dimesions for POS tags. (DEFAULT=20)',
    )
    parser.add_argument(
        '--lstm-units',
        dest='lstm_units',
        type=int,
        default=200,
        help='Number of LSTM units in each layer/direction. (DEFAULT=200)',
    )
    parser.add_argument(
        '--hidden-units',
        dest='hidden_units',
        type=int,
        default=200,
        help='Number of hidden units for each FC ReLU layer. (DEFAULT=200)',
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=50,
        help='Number of training epochs. (DEFAULT=10)',
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=10,
        help='Number of sentences per training update. (DEFAULT=10)',
    )
    parser.add_argument(
        '--droprate',
        dest='droprate',
        type=float,
        default=0.5,
        help='Dropout probability. (DEFAULT=0.5)',
    )
    parser.add_argument(
        '--unk-param',
        dest='unk_param',
        type=float,
        default=0.8375,
        help='Parameter z for random UNKing. (DEFAULT=0.8375)',
    )
    parser.add_argument(
        '--alpha',
        dest='alpha',
        type=float,
        default=1.0,
        help='Softmax distribution weighting for exploration. (DEFAULT=1.0)',
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        type=float,
        default=0,
        help='Probability of using oracle action in exploration. (DEFAULT=0)',
    )
    parser.add_argument('--np-seed', type=int, dest='np_seed')

    args = parser.parse_args()

    if args.np_seed is not None:
        import numpy as np

        np.random.seed(args.np_seed)

    print(args)

    data_dir = '/Users/qiwang/python-space/nju_nlp_tools/testdata/'
    args.model = data_dir + 'parser91.6.model'
    args.vocab = data_dir + 'toy.vocab.json'
    args.train = data_dir + 'toy.clean'
    # args.test = data_dir +'dev.clean'
    args.dev = data_dir + 'toy.clean'
    args.kbest = data_dir + 'toy.kbest'
    args.gold = data_dir + 'toy.clean'
    fm = generate_vocab(args)
    mode = 3
    sents = [('I', 'PRP'), ('do', 'MD'), ('like', 'VBP'), ('eating', 'VBG'), ('fish', 'NN')]
    if mode == 0:
        train(fm, args)
    elif mode == 1:
        test(fm, args)
    elif mode == 2:
        parse_sentence(fm, args, sents)
    else:
        rerank(fm, args)
