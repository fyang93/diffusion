#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from diffusion import Diffusion
from evaluate import OxfordParisEvaluator


def search(args, gamma=3):
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)

    time0 = time.time()
    print('[search] 1) k-NN search')
    sims, ids = diffusion.knn.search(dataset.queries, args.kq)
    sims = sims ** gamma
    qr_num = ids.shape[0]

    print('[search] 2) linear combination')
    all_scores = np.empty((qr_num, args.truncation_size), dtype=np.float32)
    all_ranks = np.empty((qr_num, args.truncation_size), dtype=np.int)
    for i in tqdm(range(qr_num), desc='[search] query'):
        scores = sims[i] @ offline[ids[i]]
        parts = np.argpartition(-scores, args.truncation_size)[:args.truncation_size]
        ranks = np.argsort(-scores[parts])
        all_scores[i] = scores[parts][ranks]
        all_ranks[i] = parts[ranks]
    print('[search] search costs {:.2f}s'.format(time.time() - time0))

    # 3) evaluation
    evaluate(all_ranks)


def evaluate(all_ranks):
    if args.ground_truth_path:
        mAP = evaluator.evaluate(all_ranks=all_ranks)
        print('mAP = {:.3f}'.format(mAP))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help="""
                        Name of the dataset
                        """)
    parser.add_argument('--query_path',
                        type=str,
                        required=True,
                        help="""
                        Path to query features
                        """)
    parser.add_argument('--gallery_path',
                        type=str,
                        required=True,
                        help="""
                        Path to gallery features
                        """)
    parser.add_argument('--ground_truth_path',
                        type=str,
                        help="""
                        Path to ground-truth
                        """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=1000,
                        help="""
                        Number of images in the truncated gallery
                        """)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)
    if args.ground_truth_path:
        evaluator = OxfordParisEvaluator(args.ground_truth_path)
    dataset = Dataset(args.query_path, args.gallery_path)
    diffusion = Diffusion(dataset.gallery, args.cache_dir)
    search(args)

