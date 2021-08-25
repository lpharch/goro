import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import random
import argparse
from tqdm import tqdm
from utils import ReplayBuffer
from agent import BQN
import torch.autograd.profiler as profiler
import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument('--lr_rate', type=float, default=1e-6, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=6, help='action scale between -1 ~ +1')

parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 1)')
args = parser.parse_args()

use_tensorboard = args.tensorboard
action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma

if use_tensorboard : 
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None
os.makedirs('./model_weights', exist_ok=True)


# env = gym.make("BipedalWalker-v3")
state_space  = 65
action_space = 19
action_scale = 6
csv_paths = "/home/ml/test/goro/csv/"
# print('observation space : ', env.observation_space)
# print('action space : ', env.action_space)
# print(env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device)
if args.load != 'no':
    agent.load_state_dict(torch.load('./model_weights/'+args.load))


memory = ReplayBuffer(200000000, action_space, device)
# real_action = np.linspace(-1.,1., action_scale)

 

def run():
    fcsvs = os.listdir(csv_paths)
    for csv in tqdm(fcsvs):
        memory.read(csv_paths+csv)
    print("reading csv file is done, loading to the buffer..")    
    memory.load()
    print("Number of entries: ", memory.size())
    print("reward Distribution: ", memory.info())
    for n_epi in range(10000000):
        done = False
        score = 0.0
        loss = agent.train_model(n_epi, memory, batch_size, gamma, use_tensorboard,writer)
        
        if(n_epi%1000==0):
            print("Loss", loss, n_epi)
            agent.save_model(n_epi)
                    
                    
print("---Running the gem5 model")
run()       
# with profiler.profile(record_shapes=True) as prof:
    # with profiler.record_function("run"):
        # run()   

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


























