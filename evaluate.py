import scipy.io as sio
import numpy as np


class OxfordParisEvaluator:
    """ Evaluate oxford/paris using official metric
    This code follows MATLAB implementation in https://github.com/ahmetius/diffusion-retrieval/blob/master/compute_map.m
    """
    def __init__(self, gt_path):
        self._anno_data = sio.loadmat(gt_path)

    def evaluate(self, all_ranks):
        aps = []
        for i in range(len(all_ranks)):
            gts_ok, gts_junk = \
                self._anno_data['gnd'][:,0][i][0][0] - 1, \
                self._anno_data['gnd'][:,0][i][1][0] - 1
            ap = self.compute_ap(all_ranks[i], gts_ok, gts_junk)
            aps.append(ap)
        return np.mean(aps)

    def compute_ap(self, ranks, qgnd, qgndj, verbose=True):
        """
        ranks: ranks of a query
        qgnd:  groundtruth image id list
        qgndj: junk image id list
        """
        pos = np.in1d(ranks, qgnd).nonzero()[0]
        junk = np.in1d(ranks, qgndj).nonzero()[0]
        pos = np.sort(pos)
        junk = np.sort(junk)

        k = 0
        ij = 0
        if qgndj.shape[0] > 0:
            ip = 0
            while ip < len(pos):
                while (ij < len(junk)) and (pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] -= k
                ip += 1

        ap = self.score_ap_from_ranks(pos, len(qgnd))
        return ap

    def score_ap_from_ranks(self, ranks, nres):
        nimgranks = len(ranks)
        ap = 0
        recall_step = 1.0 / float(nres)
        precision_0 = 1.0
        for j in range(nimgranks):
            rank = ranks[j]
            precision_1 = (j + 1) / (rank + 1);
            ap += (precision_0 + precision_1) * recall_step / 2.0
            precision_0 = precision_1
        return ap
