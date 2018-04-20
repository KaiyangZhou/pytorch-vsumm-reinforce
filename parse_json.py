import os
import argparse
import re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils import read_json

"""
Parse json file (.json) to extract rewards for specific videos.

How to use:
# image will be saved in path: blah_blah_blah
$ python parse_json.py -p blah_blah_blah/rewards.json -i 0
"""

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to rewards.json; output saved to the same dir")
parser.add_argument('-i', '--idx', type=int, default=0, help="choose which video to visualize, index starts from 0 (default: 0)")
args = parser.parse_args()

reward_writers = read_json(args.path)
keys = reward_writers.keys()
assert args.idx < len(keys)
key = keys[args.idx]
rewards = reward_writers[key]

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("{}".format(key))
plt.savefig(osp.join(osp.dirname(args.path), 'epoch_reward_' + str(args.idx) + '.png'))
plt.close()