from multiprocessing.connection import Listener
import numpy as np
import threading
import time
import pickle
from utils import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import argparse
from tqdm import tqdm
from agent import BQN
import torch.autograd.profiler as profiler
import shutil
import random
from random import randint
import zmq




parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-2, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=2, help='action scale between -1 ~ +1')
parser.add_argument("--s1", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s2", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s3", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--levels", type=int, default = 32, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.99, help = 'print interval(default : 1)')
parser.add_argument("--name", type=str, default = 'unknown')
parser.add_argument("--mlmode", type=str, default = 'training')

args = parser.parse_args()

action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
run_name = args.name
s1 = args.s1
s2 = args.s2
s3 = args.s3
leaky = args.leaky
levels = args.levels
mlmode = args.mlmode



state_space  = 28
action_space = 20
action_scale = 2
total_reward = 0

mins = [100000000] * state_space
maxs = [0.00000001] * state_space
histo = [0] * action_space


device = 'cuda' if torch.cuda.is_available() else 'cpu'



agent = BQN(state_space, action_space, action_scale, learning_rate, device)



if __name__ == "__main__":
    # creating thread
    tot_degree = 0
    mmax = 0
    mmin = 100
    actions_obs ={}
    for i in range(1000):
        state = []
        for _ in range(state_space):
            #state.append(randint(0, 1))    
            state.append(random.uniform(0, 1))    
        action = agent.action(state, True)
        
        tmp_degree = 0
        for idx, a in enumerate(action):
            tmp_degree += a
            tot_degree += a
            if(a!=0):
                histo[idx] += 1
        if(mmax<tmp_degree):
            mmax = tmp_degree
        if(mmin>tmp_degree):
            mmin = tmp_degree
        
        print("Round", i)
        print(tot_degree/(i+1), mmin, mmax)
        action_str =  ' '.join(str(e) for e in action)
        print("action", action_str)
        print("state", state)
        if(actions_obs.get(action_str)):
            r = actions_obs.get(action_str)
            actions_obs[action_str] = r + 1
        else:
            actions_obs[action_str] = 1 
        print("histo", histo)
        histo_per = []
        for idx, a in enumerate(histo):
           histo_per.append( round(a / (i+1), 2))
        print("hist %", histo_per)
        print("---- unique actions count ", len(actions_obs))
    #print(actions_obs) 
