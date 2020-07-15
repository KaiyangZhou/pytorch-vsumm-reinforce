import os
import argparse
import re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils import read_json
import numpy as np

"""
Parse json file (.json) to extract rewards for specific videos.

How to use:
# image will be saved in path: blah_blah_blah
$ python parse_json.py -p blah_blah_blah/rewards.json -i 0
"""

# Rewards in RL are typically have a high variance,
# so it's better to smooth them out for better analysis
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to rewards.json; output saved to the same dir")
parser.add_argument('-i', '--idx', type=int, default=0, help="choose which video to visualize, index starts from 0 (default: 0)")
args = parser.parse_args()

reward_writers = read_json(args.path)
keys = reward_writers.keys()
assert args.idx < len(keys)
key = keys[args.idx]
rewards = reward_writers[key]

rewards = np.array(rewards)
rewards = movingaverage(rewards, 8)

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("{}".format(key))
plt.savefig(osp.join(osp.dirname(args.path), 'epoch_reward_' + str(args.idx) + '.png'))
plt.close()