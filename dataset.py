#!/usr/bin/env python
# -*- coding: utf-8 -*-

" dataset module "

import os
import numpy as np
import joblib


def load(path):
    """Load features
    """
    if not os.path.exists(path):
        raise Exception("{} does not exist".format(path))
    ext = os.path.splitext(path)[-1]
    return {'.npy': np, '.jbl': joblib}[ext].load(path)


class Dataset(object):
    """Dataset class
    """

    def __init__(self, query_path, gallery_path):
        self.query_path = query_path
        self.gallery_path = gallery_path
        self._queries = None
        self._gallery = None

    @property
    def queries(self):
        if self._queries is None:
            self._queries = load(self.query_path)
        return self._queries

    @property
    def gallery(self):
        if self._gallery is None:
            self._gallery = load(self.gallery_path)
        return self._gallery

