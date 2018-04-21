# pytorch-vsumm-reinforce
This repo contains the Pytorch implementation of the AAAI'18 paper - [Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward](https://arxiv.org/abs/1801.00054). The original Theano implementation can be found [here](https://github.com/KaiyangZhou/vsumm-reinforce).

<div align="center">
  <img src="imgs/pipeline.jpg" alt="train" width="80%">
</div>

The main requirements are [pytorch](http://pytorch.org/) and `python 2.7`. Some dependencies that may not be installed in your machine are [tabulate](https://pypi.org/project/tabulate/) and [h5py](https://github.com/h5py/h5py). Please install other missing dependencies.

## Get started
1. Download preprocessed datasets
```bash
git clone https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
cd pytorch-vsumm-reinforce
# download datasets.tar.gz (173.5MB)
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz
tar -xvzf datasets.tar.gz
```
2. Make splits
```bash
python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets --save-name summe_splits  --num-splits 5
```
As a result, the dataset is randomly split for 5 times, which are saved as json file.

Train and test codes are written in `main.py`. To see the detailed arguments, please do `python main.py -h`.

## How to train
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
```

## How to test
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume path_to_your_model.pth.tar --verbose
```

## Plot
We provide codes to plot the rewards obtained at each epoch. Use `parse_log.py` to plot the average rewards
```bash
python parse_log.py -p path_to/log_train.txt
```
The plotted image would look like
<div align="center">
  <img src="imgs/overall_reward.png" alt="overall_reward" width="50%">
</div>

If you wanna plot the epoch-reward curve for some specific videos, do
```
python parse_json.py -p path_to/rewards.json -i 0
```

You will obtain images like
<div align="center">
  <img src="imgs/epoch_reward_0.png" alt="epoch_reward" width="30%">
  <img src="imgs/epoch_reward_13.png" alt="epoch_reward" width="30%">
  <img src="imgs/epoch_reward_15.png" alt="epoch_reward" width="30%">
</div>

If you prefer to visualize the epoch-reward curve for all training videos, try `parse_json.sh`. Modify the code according to your purpose.

## How to use your own data
We preprocess data by extracting image features for videos and save them to `h5` file. The file format looks like [this](https://github.com/KaiyangZhou/vsumm-reinforce/issues/1#issuecomment-363492711). After that, you can make split via `create_split.py`. If you wanna train policy network using the entire dataset, just do `train_keys = dataset.keys()`. [Here](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/master/main.py#L75) is the code where we initialize dataset. If you have any problems, feel free to contact me by email or raise an `issue`.

## Citation
```
@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```