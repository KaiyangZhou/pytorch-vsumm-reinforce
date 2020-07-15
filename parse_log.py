import os
import argparse
import re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

"""
Parse log file (.txt) to extract rewards.

How to use:
# image will be saved in path: blah_blah_blah
$ python parse_log.py -p blah_blah_blah/log_train.txt
"""

# Rewards in RL are typically have a high variance,
# so it's better to smooth them out for better analysis
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to log.txt; output saved to the same dir")
args = parser.parse_args()

if not osp.exists(args.path):
    raise ValueError("Given path is invalid: {}".format(args.path))

if osp.splitext(osp.basename(args.path))[-1] != '.txt':
    raise ValueError("File found does not end with .txt: {}".format(args.path))

regex_reward = re.compile('reward ([\.\deE+-]+)')
rewards = []

with open(args.path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        reward_match = regex_reward.search(line)
        if reward_match:
            reward = float(reward_match.group(1))
            rewards.append(reward)

rewards = np.array(rewards)
rewards = movingaverage(rewards, 8)

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("Overall rewards")
plt.savefig(osp.join(osp.dirname(args.path), 'overall_reward.png'))
plt.close()
