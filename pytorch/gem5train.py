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
parser.add_argument("--name", type=str, default = 'unknown')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 1)')

parser.add_argument("--s1", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s2", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s3", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--levels", type=int, default = 16, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.95, help = 'print interval(default : 1)')


args = parser.parse_args()

use_tensorboard = args.tensorboard
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
csv_paths = "/mnt2/goro_ml/goro/csv/"
# print('observation space : ', env.observation_space)
# print('action space : ', env.action_space)
# print(env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:1")
# device = 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device, s1, s2, s3, leaky).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device, s1, s2, s3, leaky)
if args.load != 'no':
    agent.load_state_dict(torch.load('./model_weights/'+args.load))


memory = ReplayBuffer(10000, action_space, device, levels)
# real_action = np.linspace(-1.,1., action_scale)

 

    
def run():
    fcsvs = os.listdir(csv_paths)
    counter = 0
    for csv in tqdm(fcsvs):
        counter +=1
        memory.read(csv_paths+csv)
        # if(counter == 100):
            # break

    print("reading csv file is done, loading to the buffer.. ", counter)    
    memory.load()
    total_reward = 0 
    for episod in range(100000):
        loss_val = 0
        reward = 0
        conf = 0
        not_conf = 1
        state, init_idx = memory.init()
        episode_reward = 0 
        bad_samples = 0
        for n_itr in range(10):
            actions = agent.action(state)

            # print(actions)
            next_state, reward, confidence = memory.step(state, actions, init_idx)
            total_reward += reward[0]
            episode_reward += reward[0]
            if(memory.size() > batch_size):
                loss = agent.train_model(n_itr, memory, batch_size, gamma, use_tensorboard,writer)
                loss_val = loss.item()
            
            if(bad_samples == 1):
                break
                
            if(confidence or bad_samples < 1):
                if(confidence):
                    conf += 1
                else:
                    bad_samples += 1
                memory.write_buffer(state, next_state, actions, reward)
            else:
                not_conf +=1
            state = next_state    
        
        
        #    if(n_itr % 10) == 0:
        output = "Epi: %r n_itr %r: tot_r:%r ep_r:%r, loss:%r conf:%r bad_samples:%r." % (episod, n_itr, total_reward,  round(episode_reward/(n_itr+1), 2), round(loss_val, 2), round(conf/((conf+not_conf)*1.0), 2), bad_samples)
        output += " As:"+' '.join([str(elem) for elem in actions])
        output += " "+ memory.info()
         # output += " "+  ' '.join([str(elem) for elem in actions])
        print(output)
        file1 = open(run_name+"_out.txt", "a")  # append mode
        file1.write(output+"\n")
        file1.close()

                
                

        if(episod % 100 == 0):    
            agent.save_model(run_name+"_"+str(episod))
        
        
        
print("---Running the gem5 model")
run()       
# with profiler.profile(record_shapes=True) as prof:
    # with profiler.record_function("run"):
        # run()   

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


























