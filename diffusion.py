#!/usr/bin/env python
# -*- coding: utf-8 -*-

" diffusion module "

import os
import time
import numpy as np
import joblib
from joblib import Parallel, delayed
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from knn import KNN, ANN


trunc_ids = None
trunc_init = None
lap_alpha = None


def get_offline_result(i):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    return scores


def cache(filename):
    """Decorator to cache results
    """
    def decorator(func):
        def wrapper(*args, **kw):
            self = args[0]
            path = os.path.join(self.cache_dir, filename)
            time0 = time.time()
            if os.path.exists(path):
                result = joblib.load(path)
                cost = time.time() - time0
                print('[cache] loading {} costs {:.2f}s'.format(path, cost))
                return result
            result = func(*args, **kw)
            cost = time.time() - time0
            print('[cache] obtaining {} costs {:.2f}s'.format(path, cost))
            joblib.dump(result, path)
            return result
        return wrapper
    return decorator


class Diffusion(object):
    """Diffusion class
    """
    def __init__(self, features, cache_dir):
        self.features = features
        self.N = len(self.features)
        self.cache_dir = cache_dir
        # use ANN for large datasets
        self.use_ann = self.N >= 100000
        if self.use_ann:
            self.ann = ANN(self.features, method='cosine')
        self.knn = KNN(self.features, method='cosine')

    @cache('offline.jbl')
    def get_offline_results(self, n_trunc, kd=50):
        """Get offline diffusion results for each gallery feature
        """
        print('[offline] starting offline diffusion')
        print('[offline] 1) prepare Laplacian and initial state')
        global trunc_ids, trunc_init, lap_alpha
        if self.use_ann:
            _, trunc_ids = self.ann.search(self.features, n_trunc)
            sims, ids = self.knn.search(self.features, kd)
            lap_alpha = self.get_laplacian(sims, ids)
        else:
            sims, ids = self.knn.search(self.features, n_trunc)
            trunc_ids = ids
            lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd])
        trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1

        print('[offline] 2) gallery-side diffusion')
        results = Parallel(n_jobs=-1, prefer='threads')(delayed(get_offline_result)(i)
                                      for i in tqdm(range(self.N),
                                                    desc='[offline] diffusion'))
        all_scores = np.concatenate(results)

        print('[offline] 3) merge offline results')
        rows = np.repeat(np.arange(self.N), n_trunc)
        offline = sparse.csr_matrix((all_scores, (rows, trunc_ids.reshape(-1))),
                                    shape=(self.N, self.N),
                                    dtype=np.float32)
        return offline

    # @cache('laplacian.jbl')
    def get_laplacian(self, sims, ids, alpha=0.99):
        """Get Laplacian_alpha matrix
        """
        affinity = self.get_affinity(sims, ids)
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    # @cache('affinity.jbl')
    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            affinity: affinity matrix
        """
        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims ** gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        return affinity
