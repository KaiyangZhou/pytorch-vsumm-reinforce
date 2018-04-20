from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import math
import numpy as np

from utils import write_json

parser = argparse.ArgumentParser("Code to create splits in json form")
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('--save-dir', type=str, default='datasets', help="path to save output json file (default: 'datasets/')")
parser.add_argument('--save-name', type=str, default='splits', help="name to save as, excluding extension (default: 'splits')")
parser.add_argument('--num-splits', type=int, default=5, help="how many splits to generate (default: 5)")
parser.add_argument('--train-percent', type=float, default=0.8, help="percentage of training data (default: 0.8)")

args = parser.parse_args()

def split_random(keys, num_videos, num_train):
    """Random split"""
    train_keys, test_keys = [], []
    rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
    for key_idx, key in enumerate(keys):
        if key_idx in rnd_idxs:
            train_keys.append(key)
        else:
            test_keys.append(key)

    assert len(set(train_keys) & set(test_keys)) == 0, "Error: train_keys and test_keys overlap"

    return train_keys, test_keys

def create():
    print("==========\nArgs:{}\n==========".format(args))
    print("Goal: randomly split data for {} times, {:.1%} for training and the rest for testing".format(args.num_splits, args.train_percent))
    print("Loading dataset from {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    keys = dataset.keys()
    num_videos = len(keys)
    num_train = int(math.ceil(num_videos * args.train_percent))
    num_test = num_videos - num_train
    print("Split breakdown: # total videos {}. # train videos {}. # test videos {}".format(num_videos, num_train, num_test))
    splits = []

    for split_idx in range(args.num_splits):
        train_keys, test_keys = split_random(keys, num_videos, num_train)
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys,
            })

    saveto = osp.join(args.save_dir, args.save_name + '.json')
    write_json(splits, saveto)
    print("Splits saved to {}".format(saveto))

    dataset.close()

if __name__ == '__main__':
    create()