'''
Mimic a continuous space attack by performing a PGD attack
in the token input embedding space.

Report fooling rate
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained sentiment classifier .th model')
    commandLineParser.add_argument('DATA_PATH', type=str, help='data filepath')
    commandLineParser.add_argument('--start_ind', type=int, default=0, help="tweet index to start at")
    commandLineParser.add_argument('--end_ind', type=int, default=100, help="tweet index to end at")
    commandLineParser.add_argument('--epsilon', type=float, default=0.01, help="l-inf pgd perturbation size")
    commandLineParser.add_argument('--lr', type=float, default=0.1, help="pgd learning rate")
    commandLineParser.add_argument('--epochs', type=int, default=20, help="Number of epochs for PGD attacks")
    commandLineParser.add_argument('--seed', type=int, default=1, help="seed for randomness")
    commandLineParser.add_argument('--cpu', type=str, default='no', help="force cpu use")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    data_path = args.DATA_PATH
    start_ind = args.start_ind
    end_ind = args.end_ind
    epsilon = args.epsilon
    lr = args.lr
    epochs = args.epochs
    seed = args.seed
    cpu_use = args.cpu

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/pgd_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')