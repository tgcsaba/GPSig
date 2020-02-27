from models import train_gpsigrnn_classifier

import sys
import os
import json

GPU_ID = str(sys.argv[1]) if len(sys.argv) > 1 else '-1'

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)

# here a grid-search is performed with using a double hold-out set on the training data
# 20% of the data is for early stopping, 20% of it is for determining the best performing architecture
# the model itself is trained in 60% of the training data

rnn_types = ['LSTM', 'GRU']

for rnn_type in rnn_types:
    
    # create results folder if not exists
    results_dir = './gridsearch/GPSig{}/'.format(rnn_type)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    # run all datasets
    for dataset in datasets:
        # use dropout or not
        for use_dropout in [True, False]:
            # choose hidden units
            for num_hidden in [8, 32, 128]:

                results_filename = os.path.join(results_dir, '{}_H{}_D{}.txt'.format(dataset, num_hidden, int(use_dropout)))

                if os.path.exists(results_filename):
                    print('{} already exists, continuing...'.format(results_filename))
                    continue

                with open(results_filename, 'w'):
                    pass

                train_gpsigrnn_classifier(dataset, num_levels=4, num_hidden=num_hidden, rnn_type=rnn_type, use_dropout=use_dropout, num_inducing=500,
                                       max_len=400, num_lags=0, increments=True, learn_weights=False, val_split=0.2, test_split=0.2, save_dir=results_dir)