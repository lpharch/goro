import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device ='cpu'
 
    

class QNetwork(nn.Module):
    def __init__(self,state_space : int, action_num : int,action_scale : int):
        super(QNetwork,self).__init__()
        self.linear_1 = nn.Linear(state_space,state_space*1)
        self.linear_2 = nn.Linear(state_space*1,state_space*1)
        # self.dropout = nn.Dropout(0.25)
        
        self.actions = [nn.Sequential(nn.Linear(state_space*1,state_space*1),
              nn.LeakyReLU(0.1),
              nn.Linear(state_space*1,action_scale)
              ) for _ in range(action_num)]

        self.actions = nn.ModuleList(self.actions)

        self.value = nn.Sequential(nn.Linear(state_space*1,state_space*1),
              nn.LeakyReLU(0.1),
              nn.Linear(state_space*1,1)
              )
        
    def forward(self,x):
        x = F.relu(self.linear_1(x))
        encoded = F.relu(self.linear_2(x))
        actions = [x(encoded) for x in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            actions[i] += value
        return actions

# class QNetwork(nn.Module):
    # def __init__(self,state_space : int, action_num : int,action_scale : int):
        # super(QNetwork,self).__init__()
        # self.linear_1 = nn.Linear(state_space,state_space*2)
        # self.linear_2 = nn.Linear(state_space*2,state_space*1)
         
        # self.actions = [nn.Sequential(nn.Linear(state_space*1,state_space*1),
              # nn.ReLU(),
              # nn.Linear(state_space*1,action_scale)
              # ) for _ in range(action_num)]
 
        # self.actions = nn.ModuleList(self.actions)

        # self.value = nn.Sequential(nn.Linear(state_space,state_space*1),
              # nn.ReLU(),
              # nn.Linear(state_space*1,1)
              # )
         
    # def forward(self,x):
        # x = F.relu(self.linear_1(x))
        # encoded = F.relu(self.linear_2(x))
        # actions = [x(encoded) for x in self.actions]
        # value = self.value(encoded)
        # for i in range(len(actions)):
            # actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            # actions[i] += value
        # return actions
        
        
state_space=16
action_num=7
action_scale=5

target_q = QNetwork(state_space, action_num, action_scale).to(device)

checkpoint = torch.load(("./gem5model_latest"), map_location=torch.device('cpu'))
target_q.load_state_dict(checkpoint['modelA_state_dict'])


def get_actions(states):
    action = target_q(torch.tensor(states).float().reshape(1,-1).to(device))
    # print("action1 ",action)
    action = [int(x.max(1)[1]) for x in action]
    # print("action2 ",action)
    # input()
    return action
    
    
    

# inp=[2.000000e+00,1.000000e+00,1.000000e+00,1.074280e+05,4.805610e+05
# ,5.949500e+04,1.896300e+04,1.074380e+05,4.639600e+04,9.203000e+04
# ,3.000000e+00,1.000000e+00,1.000000e+00,2.869800e+04,1.254120e+05
# ,2.857300e+04,9.900000e+01,2.869400e+04,6.805000e+03,1.500000e+01
# ,1.880000e+02,0.000000e+00,0.000000e+00,1.000000e+00,2.776270e+06
# ,0.000000e+00,0.000000e+00,3.720905e+06,1.999949e+06,0.000000e+00
# ,2.075000e+03,6.695900e+04,1.314500e+04,2.831339e+06,1.332200e+04
# ,2.787000e+04,7.630000e+03,4.250000e+03,1.328800e+04,9.400000e+01
# ,1.363000e+03]  
# inp=np.asarray(inp)
# inp=np.random.rand(state_space) 
# print(inp) 
# print(get_actions(np.asarray(inp))) 


# for j in range(state_space):
    # st = np.random.rand(state_space) 
    # for i in range(0, 10000000, 1000):
        # st[j]=i
        # print("actions", j, i, get_actions(st))


for i in range(0, 10000):
    st = np.random.rand(state_space) 
    print("actions", i, get_actions(st))
    
    
    
# for name, param in target_q.named_parameters():   
    # print(name, param)
    
    
    
    
    
