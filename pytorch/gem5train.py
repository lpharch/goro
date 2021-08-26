import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from utils import ReplayBuffer
from agent import BQN
import torch.autograd.profiler as profiler
import gym
from random import random
from random import randint


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
# device = 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device)
if args.load != 'no':
    agent.load_state_dict(torch.load('./model_weights/'+args.load))


memory = ReplayBuffer(10000, action_space, device)
# real_action = np.linspace(-1.,1., action_scale)

 

def run():
    fcsvs = os.listdir(csv_paths)
    counter = 0
    for csv in tqdm(fcsvs):
        counter +=1
        memory.read(csv_paths+csv)

    print("reading csv file is done, loading to the buffer.. ", counter)    
    memory.load()
    total_reward = 0 
    for episod in range(100):
        loss_val = 0
        reward = 0
        conf = 0
        not_conf = 0
        state = memory.init()
        episode_reward = 0 
        for n_itr in range(30000):
            actions = agent.action(state)
            next_state, reward, confidence = memory.step(state, actions)
            total_reward += reward[0]
            episode_reward += reward[0]
            if(memory.size() > batch_size):
                loss = agent.train_model(n_itr, memory, batch_size, gamma, use_tensorboard,writer)
                loss_val = loss.item()

            if(confidence):
                conf += 1
                memory.write_buffer(state, next_state, actions, reward)
            else:
                not_conf +=1
            state = next_state        
            if(n_itr%100==0): 
                output = "Episode: %r Iteratio:%r reward:%r total_reward:%r episode_reward:%r, loss:%r #items:%r conf. ratio:%r." % (episod, n_itr, reward[0], total_reward, episode_reward, round(loss_val, 2), memory.size(), round(conf/((conf+not_conf)*1.0), 2))
                print(output)
                file1 = open("gem5_out.txt", "a")  # append mode
                file1.write(output+"\n")
                file1.close()

            if(n_itr%1000==0): 
                print(memory.info())
                file1 = open("gem5_out.txt", "a")  # append mode
                file1.write(memory.info()+"\n")
                file1.close()
        agent.save_model(episod)
        
        
        
print("---Running the gem5 model")
run()       
# with profiler.profile(record_shapes=True) as prof:
    # with profiler.record_function("run"):
        # run()   

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


























