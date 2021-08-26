import torch
import pandas as pd
import collections
import random
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
global_rewards = np.zeros(5) 
import math
from random import randint

    
class ReplayBuffer():
    def __init__(self,buffer_limit,action_space,device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.device = device
        self.total_error = 0
        self.simulator = pd.DataFrame()
        self.state_strings = {"core0.decode.blockedCycles", "core0.fetch.cycles", "core0.numCycles", "core0.numSimulatedInsts", "core0.rename.LQFullEvents", "core0.rename.unblockCycles", "core0.rob.reads", "core0.switch_cpus0.numRate", "core0.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core0.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core0.system.switch_cpus0.issueRate", "core0.timesIdled", "core0.l2cache.ReadReq.hits::total", "core0.l2cache.demandAccesses::total", "core1.decode.blockedCycles", "core1.fetch.cycles", "core1.numCycles", "core1.numSimulatedInsts", "core1.rename.LQFullEvents", "core1.rename.unblockCycles", "core1.rob.reads", "core1.switch_cpus0.numRate", "core1.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core1.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core1.system.switch_cpus0.issueRate", "core1.timesIdled", "core1.l2cache.ReadReq.hits::total", "core1.l2cache.demandAccesses::total", "core2.decode.blockedCycles", "core2.fetch.cycles", "core2.numCycles", "core2.numSimulatedInsts", "core2.rename.LQFullEvents", "core2.rename.unblockCycles", "core2.rob.reads", "core2.switch_cpus0.numRate", "core2.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core2.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core2.system.switch_cpus0.issueRate", "core2.timesIdled", "core2.l2cache.ReadReq.hits::total", "core2.l2cache.demandAccesses::total", "core3.decode.blockedCycles", "core3.fetch.cycles", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents", "core3.rename.unblockCycles", "core3.rob.reads", "core3.switch_cpus0.numRate", "core3.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core3.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core3.system.switch_cpus0.issueRate", "core3.timesIdled", "core3.l2cache.ReadReq.hits::total", "core3.l2cache.demandAccesses::total", "core3.ReadSharedReq.mshrMisses::total", "core3.mem_ctrls.numStayReadState", "core3.mem_ctrls.rdQLenPdf::3", "core3.mem_ctrls.totGap", "core3.system.l3.ReadSharedReq.accesses::total", "core3.system.l3.demandAccesses::total", "core3.system.l3.prefetcher.prefetchersx.pfSpanPage", "core3.system.l3.tags.totalRefs", "core3.system.mem_ctrls.requestorReadAccesses::cpu0.dcache.prefetcher.prefetchers1"} 
        self.actions_string = {"Core0.L1.P0.degree", "Core0.L1.P1.degree", "Core0.L2.P0.degree", "Core0.L2.P1.degree", "Core1.L1.P0.degree", "Core1.L1.P1.degree", "Core1.L2.P0.degree", "Core1.L2.P1.degree", "Core2.L1.P0.degree", "Core2.L1.P1.degree", "Core2.L2.P0.degree", "Core2.L2.P1.degree", "Core3.L1.P0.degree", "Core3.L1.P1.degree", "Core3.L2.P0.degree", "Core3.L2.P1.degree" , "LLC.P1.degree", "LLC.P2.degree", "LLC.P0.degree"}
   
   
        # self.state_strings = {"core0.numCycles", "core0.numSimulatedInsts", "core1.numCycles", "core1.numSimulatedInsts", "core2.numCycles", "core2.numSimulatedInsts", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents"} 
        # self.actions_string = {"Core0.L1.P0.degree", "Core0.L1.P1.degree"}
        
        self.simulator_index = dict()
        
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def write_buffer(self, state, next_state, actions, reward):
        state = np.array(state)
        next_state = np.array(next_state)
        actions = np.array(actions)
        reward = np.array(reward)[0]
        self.put((state, actions, reward, next_state, 0))
        
    def read(self, path):
        df_tmp = pd.read_csv(path)
        self.simulator = self.simulator.append(df_tmp)

    def reward_gen(self, num):
        if(num >= 4):
            return int(num-4)
        else:
            return -1
        
    def print_buffer(self):
        print(self.buffer)
    
    def dot_product(self, v1, v2):
        return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

    def cosine_measure(self, v1, v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return prod / (len1 * len2)
    
    def state_similarity(self, df, state):
      length_state = len(state)
      cur_state = 0
      df["state_prod"] = 0
      state_len1 = 0
      df["state_len2"] = 0
      for i, st in enumerate(self.state_strings):
        df["state_prod"] += (state[i]*self.simulator["S_"+st])
        state_len1 += (state[i]*state[i])
        df["state_len2"] += (self.simulator["S_"+st]*self.simulator["S_"+st])
      
      df["state_sim"] = df["state_prod"] / ( math.sqrt(state_len1) * np.sqrt(df["state_len2"]) )
      
      return df["state_sim"]
    
    def action_similarity(self, df, actions):
      length_state = len(actions)
      cur_state = 0
      df["actions_prod"] = 0
      state_len1 = 0
      df["actions_len2"] = 0
      for i, st in enumerate(self.actions_string):
        st += "_Norm_ACTION"
        df["actions_prod"] += (actions[i]*self.simulator[st])
        state_len1 += (actions[i]*actions[i])
        df["actions_len2"] += (self.simulator[st]*self.simulator[st])
      
      
      df["actions_sim"] = df["actions_prod"] / ( math.sqrt(state_len1) * np.sqrt(df["actions_len2"]) )
      return df["actions_sim"]
    
    def prepare(self):
      self.simulator["reward"] = (self.simulator["NS_core0.numSimulatedInsts"]+self.simulator["NS_core1.numSimulatedInsts"]+self.simulator["NS_core2.numSimulatedInsts"]+self.simulator["NS_core3.numSimulatedInsts"])/(self.simulator["NS_core0.numCycles"]+self.simulator["NS_core1.numCycles"]+self.simulator["NS_core2.numCycles"]+self.simulator["NS_core3.numCycles"])
      self.simulator["reward"] = self.simulator["reward"].apply(self.reward_gen)
      columns = list(self.simulator)
      for col in columns:
          if(col != "Unnamed: 0"):
              val = self.simulator[col].mean()
              self.simulator[col].fillna(value=val, inplace=True)
      
      for st in self.state_strings:
          self.simulator["S_"+st] = (self.simulator["S_"+st] - self.simulator["S_"+st].min()) / (self.simulator["S_"+st].max() - self.simulator["S_"+st].min())
          self.simulator["NS_"+st] = (self.simulator["NS_"+st] - self.simulator["S_"+st].min()) / (self.simulator["NS_"+st].max() - self.simulator["NS_"+st].min())
      
      for st in self.actions_string:
          self.simulator[st+"_Norm_ACTION"] = (self.simulator[st] - self.simulator[st].min()) / (self.simulator[st].max() - self.simulator[st].min())
          
      print("***Is there any null", self.simulator.isnull().sum().sum())
      
    
      
    # This function should find the most similar state and action in the simulator and return the next state
    def step(self, state_sim, action_sim):
      df_state  = self.state_similarity(self.simulator, state_sim)
      df_action = (self.action_similarity(self.simulator, action_sim))
      df_both = pd.DataFrame()
      df_both["state"] = df_state
      df_both["action"] = df_action
      conditions = [
            (df_both['state'] > 0.95) & (df_both['action'] > 0.95),
            (df_both['state'] > 0.85) & (df_both['action'] > 0.85),
            (df_both['state'] > 0.75) & (df_both['action'] > 0.75),
            (df_both['state'] > 0.65) & (df_both['action'] > 0.65),
            (df_both['state'] >= 0)   & (df_both['action'] >= 0),
            ]

        # create a list of the values we want to assign for each condition
      values = [0, 1, 2, 3, 4]

        # create a new column and use np.select to assign values to it using our lists as arguments
      df_both['tier'] = np.select(conditions, values)
      idx = df_both['tier'].idxmax(axis=1)
      idx_val = df_both['tier'].iloc[idx]
      print("idx val ", idx, idx_val)
      print(df_both.head)
      

      
      if(idx in self.simulator_index):
          self.simulator_index[idx] += 1
      else:
          self.simulator_index[idx] = 1
      

      next_state = []
      reward = []
      for st in self.state_strings:
          next_state.append(self.simulator["NS_"+st].iloc[idx])
      reward.append(self.simulator["reward"].iloc[idx])
      confidence_action = self.simulator["actions_sim"].iloc[idx]
      confidence_state = self.simulator["state_sim"].iloc[idx]
      confidence_sum = confidence_state + confidence_action
      
      conf = False
      if(idx_val<3):
          conf = True

      return next_state, reward, conf
    
    def init(self):
      idx = len(self.simulator)
      idx = randint(0, idx)
      state = []
      for st in self.state_strings:
          state.append(self.simulator["S_"+st].iloc[idx])
      state = np.array(state)
      return state

    
    def load(self):
      self.prepare()
      for index, row in tqdm(self.simulator.iterrows()):
          state = []
          next_state = []
          actions = []
          reward = []
          for st in self.state_strings:
              state.append((row["S_"+st]))
              next_state.append((row["NS_"+st]))
          for st in self.actions_string:
              actions.append(row[st])
          reward.append(row["reward"])
          global_rewards[reward] += 1
                    
       
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
        
    def info(self):
        out = ""
        tot = 0
        tot_high_1 = 0
        tot_high_2 = 0
        tot_high_3 = 0
        max_idx = 0
        max_val = 0
        for key in self.simulator_index:
            tot += self.simulator_index[key]
            if(self.simulator_index[key] > 1):
                tot_high_1 += self.simulator_index[key]
            if(self.simulator_index[key] > 2):
                tot_high_2 += self.simulator_index[key]
            if(self.simulator_index[key] > 3):
                tot_high_3 += self.simulator_index[key]
            if(self.simulator_index[key] > max_val):
                max_val = self.simulator_index[key]                  
                max_idx = key
                
        # for key in self.simulator_index:
            # if(self.simulator_index[key] > 1):
                # out += ("["+str(key)+"]:%"+str((self.simulator_index[key])/ (tot_high*1.0))+" ")
        
        
        out = "Total Entries: "+ str(tot)
        out += " More than 1: "+ str(round(tot_high_1/(tot*1.0), 2))
        out += " More than 2: "+ str(round(tot_high_2/(tot*1.0), 2))
        out += " More than 3: "+ str(round(tot_high_3/(tot*1.0), 2))
        out += " Hottest index: "+ str(max_idx)
        self.simulator.iloc[max_idx].to_csv("hottest.csv", mode='a')
        self.simulator.iloc[max_idx+1].to_csv("hottest.csv", mode='a')
        return out