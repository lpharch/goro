import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random
from random import randint
from network import QNetwork


 
class BQN(nn.Module):
    def __init__(self,state_space : int, action_num : int,action_scale : int, learning_rate, device : str):
        super(BQN,self).__init__()
        self.device = device
        self.q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
            
        self.optimizer = optim.AdamW([\
                                    {'params' : self.q.linear_1.parameters(), 'weight_decay':1e-4,'lr': learning_rate / (action_num+2)},\
                                    # {'params' : self.q.linear_2.parameters(), 'weight_decay':0.001,'lr': learning_rate / (action_num+2)},\
                                    {'params' : self.q.value.parameters(), 'weight_decay':1e-4, 'lr' : learning_rate/ (action_num+2)},\
                                    {'params' : self.q.actions.parameters(), 'weight_decay':1e-4, 'lr' : learning_rate},\
                                    ])  
        
        self.update_freq = 1000
        self.update_count = 0
        self.action_count = 0
        print("Loading the model")
        # self.load_model("./models/model", self.device)

    
    def action(self,x, go_low):
        acc = []
        th1 = 0.05
        toss = random()
        if(toss< th1):
            for pf in range(20):
                acc.append(randint(0, 1))
        else:
            out =  self.q(torch.tensor(x, dtype=torch.float).to(self.device))
            for tor in out:
                acc.append(torch.argmax(tor, dim=1)[[0]].item() )

        
            
        return acc
    
    def save_model(self, name):
        torch.save({
                    'modelA_state_dict': self.q.state_dict(),
                    'modelB_state_dict': self.target_q.state_dict(),
                    'optimizerA_state_dict': self.optimizer.state_dict()
                    }, "./models/"+str(name))
        # torch.save({
                    # 'modelA_state_dict': self.q.state_dict(),
                    # 'modelB_state_dict': self.target_q.state_dict(),
                    # 'optimizerA_state_dict': self.optimizer.state_dict()
                    # }, "./gem5model_latest")
    
    def load_model(self, name, device):
        checkpoint = torch.load(name, map_location=device)
        print("Trying to load the model")
        self.q.load_state_dict(checkpoint['modelA_state_dict'])
        print("model loaded")
        
    def train_model(self, memory, batch_size, gamma):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        
       
        actions = torch.stack(actions).transpose(0,1).unsqueeze(-1)
        done_mask = torch.abs(done_mask-1)

        cur_actions = self.q(state.float())

        cur_actions = torch.stack(cur_actions).transpose(0,1)
        cur_actions = cur_actions.gather(2, actions.long()).squeeze(-1)
        target_cur_actions = self.target_q(next_state.float())

        target_cur_actions = torch.stack(target_cur_actions).transpose(0,1)
        target_cur_actions = target_cur_actions.max(-1,keepdim = True)[0]

        target_action = (done_mask * gamma * target_cur_actions.mean(1).float() + reward.float())

        loss = F.smooth_l1_loss(cur_actions, target_action.repeat(1, 20))
       
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if (self.update_count % self.update_freq == 0) and (self.update_count > 0):
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())
            # print("q copied to target_q")
            
  
        return loss
