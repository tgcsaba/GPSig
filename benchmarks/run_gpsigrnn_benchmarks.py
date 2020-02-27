from models import train_gpsigrnn_classifier

import sys
import os
import json

GPU_ID = str(sys.argv[1]) if len(sys.argv) > 1 else '-1'

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)

with open('./architectures.json', 'r') as f:
    architectures = json.load(f)

rnn_types = ['LSTM' , 'GRU']
num_experiments = 5

for i in range(num_experiments):
    for rnn_type in rnn_types:
        
        model = 'Sig{}'.format(rnn_type)
        
        # create results folder if not exists
        results_dir = './results/GP{}/'.format(model)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
            
        # run all datasets
        for dataset in datasets:
            # select architecture
            if dataset not in architectures[model]:
                print('Warning: architecture missing for model : {}, dataset : {}, continuing...'.format(model, dataset))
                continue
                
            num_hidden, use_dropout = architectures[model][dataset]['H'], architectures[model][dataset]['D']

            results_filename = os.path.join(results_dir, '{}_H{}_D{}_{}.txt'.format(dataset, num_hidden, int(use_dropout), i))

            if os.path.exists(results_filename):
                print('{} already exists, continuing...'.format(results_filename))
                continue

            with open(results_filename, 'w'):
                pass

            train_gpsigrnn_classifier(dataset, num_levels=4, num_hidden=num_hidden, rnn_type=rnn_type, use_dropout=use_dropout, num_inducing=500, 
                                      max_len=500, num_lags=0, increments=True, learn_weights=False, val_split=0.2, experiment_idx=i, save_dir=results_dir)