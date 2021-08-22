import torch

import collections
import random
import numpy as np


global_rewards = np.zeros(5) 

class ReplayBuffer():
    def __init__(self,buffer_limit,action_space,device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.device = device
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def print_buffer(self):
        print("-------")
        print(self.buffer)
    
    def load(self, path):
      with open(path, 'r') as f2:
            datas = f2.read() 
            datas = datas.split("\n")
            idx=0
            for data in datas:
                if(data):
                    data = data.split(",")
                    action = data[0:7]
                    action = [float(i) for i in action] 
                    action = [int(i) for i in action] 
                    
                    s = np.asarray(data[7:48])
                    state = np.take(s, [3,5,7,8,13,14,15,17,18,31,32,34,35,36,38,39])
                    state=state.astype(np.float)
                    # state=np.log(state, where=0<state, out=abs(0*state))
                    # state=np.clip(state,0,0.1)
                     
                    
                    all_next_state = np.asarray(data[48:89])
                    all_next_state = all_next_state.astype(np.float)
                    
                    s = np.asarray(data[48:89])
                    next_state = np.take(s, [3,5,7,8,13,14,15,17,18,31,32,34,35,36,38,39])
                    next_state=next_state.astype(np.float)
                    
                    
                    # next_state=np.clip(next_state,0,10)
                    # next_state=np.log(next_state, where=0<next_state, out=abs(0*next_state))
                    # print("next_state", next_state) 
                    
                    reward = all_next_state[24]
                    # print("reward", reward)
                    # if(next_state[24]>0):
                        # reward = -1*np.log10(next_state[24]/100)+4
                        
                        # reward = 1000000/next_state[24]
                    done = 0 
                    # if(next_state[24]<1000000):
                        # reward=1
                    # reward = (int)(-1*reward/10) 
                    # global global_rewards
                    if(reward<20):
                        reward=4
                    elif(reward<30):
                        reward=3
                    elif(reward<50):
                        reward=2
                    elif(reward<70):
                        reward=1
                    else:
                        reward=0

                    global_rewards[reward]+=1
                    
                    # print("next_state", type(next_state), len(next_state), type(next_state[0]))
                    # print("reward", type(reward))
                    # print("done", type(done))
                    # print("action", type(action), len(action), type(action[0]))
                    # input()
                    self.put((state,action,reward,next_state, done))
                    
                    
      print("self.buffer", len(self.buffer))
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for i in range(self.action_space)]

        for transition in mini_batch:
            state, actions,reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(self.device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
               actions_lst ,torch.tensor(reward_lst).to(self.device),\
                torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
               torch.tensor(done_mask_lst).to(self.device)
    def size(self):
        return len(self.buffer)
        
    def info(self):
        print("Reward info ", global_rewards)