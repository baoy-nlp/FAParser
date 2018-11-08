# -*- coding:utf-8 -*-
# Filename: utility.py
# Author：hankcs
# Date: 2017-11-03 22:05
from __future__ import print_function

import errno
import sys
from os import makedirs


def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def eprint(*args, **kwargs):
    print(args)
    # print(*args, file=sys.stderr, **kwargs)


def combine_files(fids, out, tb):
    print('%d files...' % len(fids))
    total_sentence = 0
    for n, file in enumerate(fids):
        if n % 10 == 0 or n == len(fids) - 1:
            print("%c%.2f%%\r" % (13, (n + 1) / float(len(fids)) * 100), end='')
        sents = tb.parsed_sents(file)
        for s in sents:
            out.write(s.pformat(margin=sys.maxsize))
            out.write(u'\n')
            total_sentence += 1
    print()
    print('%d sentences.' % total_sentence)
    print()
