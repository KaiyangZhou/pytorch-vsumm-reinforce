import os, argparse, re
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-log', type=str, help="path to log.txt; output saved to the same dir")
args = parser.parse_args()

if not osp.exists(args.log):
    raise ValueError("Given path is invalid: {}".format(args.log))

if osp.splitext(osp.basename(args.log))[-1] != '.txt':
    raise ValueError("File found does not end with .txt: {}".format(args.log))

regex_reward = re.compile('reward ([\.\deE+-]+)')
rewards = []

with open(args.log, 'r') as f:
    lines = f.readlines()
    for line in lines:
        reward_match = regex_reward.search(line)
        if reward_match:
            reward = float(reward_match.group(1))
            rewards.append(reward)

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig(osp.join(osp.dirname(args.log), 'epoch_reward.png'))
plt.close()
