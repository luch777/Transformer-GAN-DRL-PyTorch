import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--lr_gen', type=float, default=0.00001)
parser.add_argument('--lr_dis', type=float, default=0.00005)
parser.add_argument('--gamma_g', type=float, default=0.99)
parser.add_argument('--gamma_d', type=float, default=0.95)
parser.add_argument('--lr_rl', type=float, default=0.0001)
parser.add_argument('--n_critic', type=int, default=3)
parser.add_argument('--max_iteration_num', type=int, default=15000)
parser.add_argument('--max_epoch', type=int, default=3000)
parser.add_argument('--episode_num', type=int, default=300)
parser.add_argument('--max_episode_len', type=int, default=3)
parser.add_argument('--phi', type=int, default=1)
parser.add_argument('--beta1', type=int, default="0")
parser.add_argument('--beta2', type=float, default="0.99")
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--select_num', type=int, default=30)
parser.add_argument('--gen_num_per_class', type=int, default=1500)
parser.add_argument('--n_sample_per_state', type=int, default=100)
parser.add_argument('--data_dim', type=int, default=11)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--name', type=str, default='ahu')

args = parser.parse_args()

