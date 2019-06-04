import os
import argparse
import re
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

"""
    Parse log file (.txt) to extract rewards.
    
    How to use:
        # image will be saved in path
        $ python parse_log.py -p log/summe-split0/log_train.txt
"""

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="path to log.txt; output saved to the same dir")
args = parser.parse_args()

if not os.path.exists(args.path):
    raise ValueError("Given path is invalid: {}".format(args["path"]))

if os.path.splitext(os.path.basename(args.path))[-1] != '.txt':
    raise ValueError("File found dose not end with .txt: {}".format(args.path))

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
plt.xlabel("epoch")
plt.ylabel("reward")
plt.title("Overall rewards")
plt.savefig(os.path.join(os.path.dirname(args.path), 'overall_reward.png'))
plt.close()