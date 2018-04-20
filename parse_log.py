import os
import argparse
import re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""
Parse log file (.txt) to extract rewards.

How to use:
# image will be saved in path: blah_blah_blah
$ python parse_log.py -p blah_blah_blah/log_train.txt
"""

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

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("Overall rewards")
plt.savefig(osp.join(osp.dirname(args.path), 'overall_reward.png'))
plt.close()
