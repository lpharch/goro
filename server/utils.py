import torch
import pandas as pd
import collections
import random
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
global_rewards = np.zeros(10) 
import math
from random import randint
import csv
from matplotlib import pyplot as plt
import time  
from scipy import spatial
from scipy.spatial.distance import cdist
# scipy.spatial.distance.cdist,
import heapq

class ReplayBuffer():
    def __init__(self,buffer_limit,action_space,device, levels):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.buffer_limit = buffer_limit
        self.device = device
        
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def clear_buf(self):
        self.buffer = collections.deque(maxlen=self.buffer_limit)
    
        
    def write_buffer(self, state, next_state, actions, reward):
        state = np.array(state)
        next_state = np.array(next_state)
        actions = np.array(actions)
        reward = np.array(reward)[0]
        # print("Writing to buffer", state, next_state, actions, reward)
        self.put((state, actions, reward, next_state, 0))
        
            
    def print_buffer(self):
        print(self.buffer)
    
      
       
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst, actions_lst = [], [], [], [], []
        actions_lst = [[] for i in range(self.action_space)]
        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(self.device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
               actions_lst ,\
               torch.tensor(reward_lst).to(self.device),\
               torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
               torch.tensor(done_mask_lst).to(self.device)
    def size(self):
        return len(self.buffer)
        
    