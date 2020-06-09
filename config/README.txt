Configuration README
====================

1. example config to train
# Dataset options
DATASET = 'datasets/eccv16_dataset_summe_google_pool5.h5'
SPLIT = 'datasets/summe_splits.json'
SPLIT_ID = 0
METRIC = 'summe'

# Misc
GPU = '0'
EVALUATE = False
TEST = False
VERBOSE = True
SAVE_DIR = 'log/summe-split0'

2. example config to evaluate
# Dataset options
DATASET = 'datasets/eccv16_dataset_summe_google_pool5.h5'
SPLIT = 'datasets/summe_splits.json'
SPLIT_ID = 0
METRIC = 'summe'

# Misc
GPU = '0'
EVALUATE = True
TEST = False
RESUME = 'log/summe-split0/model_epoch60.pth.tar'
VERBOSE = True
SAVE_DIR = 'log/summe-split0'
SAVE_RESULT = True

3. example config to test