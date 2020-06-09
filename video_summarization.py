from __future__ import print_function
import os
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from config import config
from utils.file_process import Logger, read_json, write_json, save_checkpoint
from networks.DSN import DSN
from networks.RL import compute_reward
from utils import vsum_tool

torch.manual_seed(config.SEED)
os.environ["CUDA_VISIBLE_DEVCIES"] = config.GPU
use_gpu = torch.cuda.is_available()
if config.USE_CPU: use_gpu = False

def main():
    if not config.EVALUATE:
        sys.stdout = Logger(os.path.join(config.SAVE_DIR, 'log_train.txt'))
    else:
        sys.stdout = Logger(os.path.join(config.SAVE_DIR, 'log_test.txt'))


    if use_gpu:
        print("Currently using GPU {}".format(config.GPU))
        cudnn.benchmark = True
        torch.cuda.manual_seed(config.SEED)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(config.DATASET))
    dataset = h5py.File(config.DATASET, 'r')
    num_videos = len(dataset.keys())

    splits = read_json(config.SPLIT)

    if not config.TEST:
        assert config.SPLIT_ID < len(splits), "split_id (got {}) exceeds {}".format(config.SPLIT_ID, len(splits ))
        split = splits[config.SPLIT_ID]
        train_keys = split["train_keys"]
        test_keys = split["test_keys"]
        print("# total videos {}. # train videos {}. # test videos {}.".format(num_videos, len(train_keys), len(test_keys)))

    print("Initialize model")
    model = DSN(in_dim=config.INPUT_DIM, hid_dim=config.HIDDEN_DIM, num_layers = config.NUM_LAYERS, cell=config.RNN_CELL)
    print("Model Size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    if config.STEP_SIZE > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size= config.STEP_SIZE, gamma=config.GAMMA)

    if config.RESUME:
        print("Loading checkpoint from '{}'".format(config.RESUME))
        checkpoint = torch.load(config.RESUME)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if config.TEST:
        print("Test only")
        test(model, dataset, ['video_1'], use_gpu)
        return


    # Evaluate
    if config.EVALUATE:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    # Train
    print("===> Start training")
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video

    for epoch in range(start_epoch, config.MAX_EPOCH):
        indices = np.arange(len(train_keys))
        np.random.shuffle(indices)

        # Input each Video to Model
        for idx in indices:
            key = train_keys[idx]
            seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)

            if use_gpu: seq = seq.cuda()
            probs = model(seq) # output shape (1, seq_len, 1)

            cost = config.BETA * (probs.mean() - 0.5) ** 2 # minimize summary length penalty term [Eq.11]
            m = Bernoulli(probs)

            epis_rewards = []
            for _ in range(config.NUM_EPISODE):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward = compute_reward(seq, actions, use_gpu=use_gpu)

                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))

        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, config.MAX_EPOCH, epoch_reward))

    write_json(reward_writers, os.path.join(config.SAVE_DIR, 'rewards.json'))
    evaluate(model, dataset, test_keys, use_gpu)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    model_save_path = os.path.join(config.SAVE_DIR, 'model_epoch' + str(config.MAX_EPOCH) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    print("Model saved to {}".format(model_save_path))

    dataset.close()


def evaluate(model, dataset, test_keys, use_gpu):
    print("===> Evaluation")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if config.METRIC == 'tvsum' else 'max'

        if config.VERBOSE: table = [["No.", "Video", "F-Score"]]

        if config.SAVE_RESULTS:
            h5_res = h5py.File(os.path.join(config.SAVE_DIR, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)

            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tool.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tool.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)


            if config.VERBOSE:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if config.SAVE_RESULTS:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if config.VERBOSE:
        print(tabulate(table))

    if config.SAVE_RESULTS: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-Score {:.1%}".format(mean_fm))

    return mean_fm

def test(model, dataset, test_data, use_gpu):
    print("===> Test")
    with torch.no_grad():
        model.eval()

        if config.SAVE_RESULTS:
            h5_res = h5py.File(os.path.join(config.SAVE_DIR, 'result_test.h5'),'w')

        for key_idx, key in enumerate(test_data):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)

            if use_gpu: seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][...]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]

            machine_summary = vsum_tool.generate_summary(probs, cps, num_frames, nfps,positions)

            if config.SAVE_RESULTS:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)

        if config.SAVE_RESULTS:
            h5_res.close()

if __name__ == '__main__':
    main()





