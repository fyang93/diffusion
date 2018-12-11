#!/usr/bin/env python
# -*- coding: utf-8 -*-

" convert .mat file to .npy file "

import os
import argparse
import numpy as np
import joblib
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        choices=['oxford5k', 'oxford105k',
                                 'paris6k', 'paris106k'],
                        help="""
                        Name of the dataset
                        """)
    parser.add_argument('--feature_type',
                        type=str,
                        required=True,
                        choices=['resnet', 'siamac'],
                        help="""
                        Feature type
                        """)
    parser.add_argument('--mat_dir',
                        type=str,
                        required=True,
                        help="""
                        Directory to the .mat file
                        """)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    input_file = '{}_{}.mat'.format(args.dataset_name, args.feature_type)
    glob_output_file = '{}_{}_glob.npy'.format(args.dataset_name, args.feature_type)
    query_dir = os.path.join(args.mat_dir, 'query')
    gallery_dir = os.path.join(args.mat_dir, 'gallery')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)
    with h5py.File(os.path.join(args.mat_dir, input_file), 'r') as f:
        glob_q = np.array([f[x[0]][:] for x in f['/glob/Q']])
        np.save(os.path.join(args.mat_dir, 'query',
                             glob_output_file), np.squeeze(glob_q, axis=1))
        glob_g = np.array([f[x[0]][:] for x in f['/glob/V']])
        np.save(os.path.join(args.mat_dir, 'gallery',
                             glob_output_file), np.squeeze(glob_g, axis=1))
