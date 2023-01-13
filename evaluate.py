import pandas as pd
from utils.eval_utils import eval_from_csv
import yaml
import argparse
import os
import os.path as osp
import json


if __name__ == '__main__':

    if not os.path.exists('results'):
        os.mkdir('results')

    # choose which results to evaluate
    possible_results_folder_names = os.listdir('results')
    possible_results_folder_names = [osp.join('results',x) for x in possible_results_folder_names]

    parser = argparse.ArgumentParser()
    parser.add_argument("-R","--results_folder_path", 
                        required=True,
                        type=str, 
                        choices=possible_results_folder_names,
                        help='Path to results folder')
    args = parser.parse_args()

    # parse choice
    results_path = args.results_folder_path
    f = open (osp.join(results_path,'options.json'), "r")
    options = json.loads(f.read())
    dataset_name = options['DATASET-NAME']


    # load dataset variables
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    if 'FP' in dataset_name:
        config = config['DATASET-VARS']['FP']
        THR_ROT = config['THR-ROT']
        THR_TRANS = config['THR-TRANS']
        NR_EXAMPLES = config[dataset_name]['N']
    else:
        config = config['DATASET-VARS'][dataset_name]
        THR_ROT = config['THR-ROT']
        THR_TRANS = config['THR-TRANS']
        NR_EXAMPLES = config['NR-EXAMPLES']

    results_df = pd.read_csv(osp.join(results_path,'results.csv'))

    eval_from_csv(data=results_df,
                  thr_rot= THR_ROT,
                  thr_trans= THR_TRANS,
                  M = NR_EXAMPLES)