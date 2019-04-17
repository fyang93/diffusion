#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print


def search():
    n_query = len(queries)
    diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    scores = features[:n_query] @ features[n_query:].T
    ranks = np.argsort(-scores.todense())
    evaluate(ranks)


def search_old(gamma=3):
    diffusion = Diffusion(gallery, args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)

    time0 = time.time()
    print('[search] 1) k-NN search')
    sims, ids = diffusion.knn.search(queries, args.kq)
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


def evaluate(ranks):
    gnd_name = os.path.splitext(os.path.basename(args.gnd_path))[0]
    with open(args.gnd_path, 'rb') as f:
        gnd = pickle.load(f)['gnd']
    compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)


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
    parser.add_argument('--gnd_path',
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
    dataset = Dataset(args.query_path, args.gallery_path)
    queries, gallery = dataset.queries, dataset.gallery
    search()

