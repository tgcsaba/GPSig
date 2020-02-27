from models import train_gpsig_classifier

import sys
import os
import json

GPU_ID = str(sys.argv[1]) if len(sys.argv) > 1 else '-1'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)

results_dir = './results/GPSig/'
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

num_experiments = 5

for i in range(num_experiments):
    for dataset in datasets:

        results_filename = os.path.join(results_dir, '{}_{}.txt'.format(dataset, i))

        if os.path.exists(results_filename):
            print('{} already exists, continuing...'.format(results_filename))
            continue

        with open(results_filename, 'w'):
            pass

        train_gpsig_classifier(dataset, num_levels=4, num_inducing=500, max_len=500, num_lags=1, increments=True, learn_weights=False,
                               val_split=0.2, experiment_idx=i, save_dir=results_dir)      