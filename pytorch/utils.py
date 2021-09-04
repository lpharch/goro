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

from matplotlib import pyplot as plt
    
class ReplayBuffer():
    def __init__(self,buffer_limit,action_space,device, levels):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.device = device
        self.total_error = 0
        self.tier_hist = np.zeros(10)
        self.simulator = pd.DataFrame()
        self.state_strings = ["core0.decode.blockedCycles", "core0.fetch.cycles", "core0.numCycles", "core0.numSimulatedInsts", "core0.rename.LQFullEvents", "core0.rename.unblockCycles", "core0.rob.reads", "core0.switch_cpus0.numRate", "core0.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core0.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core0.system.switch_cpus0.issueRate", "core0.timesIdled", "core0.l2cache.ReadReq.hits::total", "core0.l2cache.demandAccesses::total", "core1.decode.blockedCycles", "core1.fetch.cycles", "core1.numCycles", "core1.numSimulatedInsts", "core1.rename.LQFullEvents", "core1.rename.unblockCycles", "core1.rob.reads", "core1.switch_cpus0.numRate", "core1.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core1.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core1.system.switch_cpus0.issueRate", "core1.timesIdled", "core1.l2cache.ReadReq.hits::total", "core1.l2cache.demandAccesses::total", "core2.decode.blockedCycles", "core2.fetch.cycles", "core2.numCycles", "core2.numSimulatedInsts", "core2.rename.LQFullEvents", "core2.rename.unblockCycles", "core2.rob.reads", "core2.switch_cpus0.numRate", "core2.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core2.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core2.system.switch_cpus0.issueRate", "core2.timesIdled", "core2.l2cache.ReadReq.hits::total", "core2.l2cache.demandAccesses::total", "core3.decode.blockedCycles", "core3.fetch.cycles", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents", "core3.rename.unblockCycles", "core3.rob.reads", "core3.switch_cpus0.numRate", "core3.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core3.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core3.system.switch_cpus0.issueRate", "core3.timesIdled", "core3.l2cache.ReadReq.hits::total", "core3.l2cache.demandAccesses::total", "core3.ReadSharedReq.mshrMisses::total", "core3.mem_ctrls.numStayReadState", "core3.mem_ctrls.rdQLenPdf::3", "core3.mem_ctrls.totGap", "core3.system.l3.ReadSharedReq.accesses::total", "core3.system.l3.demandAccesses::total", "core3.system.l3.prefetcher.prefetchersx.pfSpanPage", "core3.system.l3.tags.totalRefs", "core3.system.mem_ctrls.requestorReadAccesses::cpu0.dcache.prefetcher.prefetchers1"]
        self.actions_string = ["Core0.L1.P0.degree", "Core0.L1.P1.degree", "Core0.L2.P0.degree", "Core0.L2.P1.degree", "Core1.L1.P0.degree", "Core1.L1.P1.degree", "Core1.L2.P0.degree", "Core1.L2.P1.degree", "Core2.L1.P0.degree", "Core2.L1.P1.degree", "Core2.L2.P0.degree", "Core2.L2.P1.degree", "Core3.L1.P0.degree", "Core3.L1.P1.degree", "Core3.L2.P0.degree", "Core3.L2.P1.degree" , "LLC.P1.degree", "LLC.P2.degree", "LLC.P0.degree"]
   
   
        # self.state_strings = {"core0.numCycles", "core0.numSimulatedInsts", "core1.numCycles", "core1.numSimulatedInsts", "core2.numCycles", "core2.numSimulatedInsts", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents"} 
        # self.actions_string = {"Core0.L1.P0.degree", "Core0.L1.P1.degree"}
        
        self.simulator_index = dict()
        self.max_val = []
        self.min_val = []
        self.step_val = []
        self.levels = levels 
        self.bins = {} 
        
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
        self.simulator = self.simulator.append(df_tmp, ignore_index=True)

    def reward_gen(self, num):
        num_q = int(num)
        # print(num_q, num)
        return num_q
            
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
      # print("-------")
      for i, st in enumerate(self.state_strings):
        # print(self.simulator["S_"+st].head())
        # print("state[i]", state[i])
        df["state_prod"] += (state[i]*self.simulator["S_"+st])
        # print(df["state_prod"].head())
        state_len1 += (state[i]*state[i])
        # print(state_len1)
        df["state_len2"] += (self.simulator["S_"+st]*self.simulator["S_"+st])
        # print(df["state_len2"].head())
        # print("---")
        # input()
      
      df["state_sim"] = df["state_prod"] / ( math.sqrt(state_len1) * np.sqrt(df["state_len2"]) )
      
      return df["state_sim"]
    
    def action_similarity(self, df, actions_sent):
      actions = []
      for i in range(len(actions_sent)):
        actions.append(actions_sent[i]+0.000001)
        
      length_state = len(actions)
      cur_state = 0
      df["actions_prod"] = 0
      state_len1 = 0
      df["actions_len2"] = 0
      for i, st in enumerate(self.actions_string):
        # st += "_Norm_ACTION"
        df["actions_prod"] += (actions[i]*(self.simulator[st]+0.000001))
        state_len1 += (actions[i]*actions[i])
        df["actions_len2"] += ((self.simulator[st]+0.000001)*(self.simulator[st]+0.000001))
      
      df["actions_sim"] = df["actions_prod"] / ( math.sqrt(state_len1) * np.sqrt(df["actions_len2"]) )
      return df["actions_sim"]
      
    def histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                         np.arange(npt),
                         np.sort(x))
  
    
    def prepare(self):
      self.simulator["IPC_now"] = (self.simulator["NS_core0.numSimulatedInsts"]/self.simulator["NS_core0.numCycles"]) + (self.simulator["NS_core1.numSimulatedInsts"]/self.simulator["NS_core1.numCycles"]) + \
                                  (self.simulator["NS_core2.numSimulatedInsts"]/self.simulator["NS_core2.numCycles"]) + (self.simulator["NS_core3.numSimulatedInsts"]/self.simulator["NS_core3.numCycles"])
      
      self.simulator["IPC_then"] = (self.simulator["S_core0.numSimulatedInsts"]/self.simulator["S_core0.numCycles"]) + (self.simulator["S_core1.numSimulatedInsts"]/self.simulator["S_core1.numCycles"]) + \
                                  (self.simulator["S_core2.numSimulatedInsts"]/self.simulator["S_core2.numCycles"]) + (self.simulator["S_core3.numSimulatedInsts"]/self.simulator["S_core3.numCycles"])
      
      self.simulator["reward"] = self.simulator["IPC_now"] - self.simulator["IPC_then"]
      
      
      columns = list(self.simulator)
      for col in columns:
          if(col != "Unnamed: 0"):
              val = self.simulator[col].mean()
              self.simulator[col].fillna(value=val, inplace=True)

      
      # for col in columns:
          # if(col != "Unnamed: 0"):
              # val = self.simulator[col].mean()
              # self.simulator[col].fillna(value=val, inplace=True)
      self.simulator["reward"] = self.simulator["reward"].apply(self.reward_gen)
      mmin = self.simulator["reward"].min()
      mmax = self.simulator["reward"].max()
      
      print("mmin", mmin)
      print("mmax", mmax)
      # hist = self.simulator["reward"](bins=mmax-mmin+1)
      # hist = self.simulator.hist(column='reward')
      # print("hist")
      
      
      out_max=""
      out_min=""
      for st in self.state_strings:
          out_max += str(self.simulator["S_"+st].max())+ ", "
          out_min += str(self.simulator["S_"+st].min())+ ", "
      print(out_max)
      print(out_min)
      
      levels = self.levels
      for st in self.state_strings:
          bins = np.zeros(levels)
          the_min = self.simulator["S_"+st].min()
          the_max = self.simulator["S_"+st].max()
          the_step = (the_max - the_min)/levels
          self.max_val.append(the_min)
          self.min_val.append(the_max)
          self.step_val.append(the_step)
          
          # for i in range(levels):
            # bins[i] = (the_min)+ i*the_step
          bins = self.histedges_equalN(self.simulator["S_"+st], levels)
          self.bins["S_"+st] = bins
          self.simulator["S_"+st] = np.digitize(self.simulator["S_"+st], bins)
          
          bins = np.zeros(levels)
          the_min = self.simulator["NS_"+st].min()
          the_max = self.simulator["NS_"+st].max()
          the_step = (the_max - the_min)/levels
          # for i in range(levels):
            # bins[i] = int(the_min)+ i*the_step
          bins = self.histedges_equalN(self.simulator["NS_"+st], levels)
          self.simulator["NS_"+st] = np.digitize(self.simulator["NS_"+st], bins)
          
      print("***Is there any null", self.simulator.isnull().sum().sum())
      
      
      for st in self.state_strings:
          print("Histogram for S_", st)
          histogram = self.simulator["S_"+st].plot.hist()
          # plt.show()
          plt.savefig("./pdfs/"+"S_"+st+".pdf")  
          plt.clf()
          plt.cla()
          plt.close()
      for st in self.state_strings:
          print("Histogram for NS_", st)
          histogram = self.simulator["NS_"+st].plot.hist()
          # plt.show()
          plt.savefig("./pdfs/"+"NS_"+st+".pdf")  
          plt.clf()
          plt.cla()
          plt.close()
      for st in self.actions_string:    
          print("Action: ", st)
          histogram = self.simulator[st].plot.hist()
          # plt.show()
          plt.savefig("./pdfs/"+st+".pdf")  
          plt.clf()
          plt.cla()
          plt.close()
        
    def min_max(self, x):
        return pd.Series(index=['min','max'],data=[x.min(),x.max()])
      
    def norm_state(self, state):
        levels = self.levels
        norm = []
        # for s_idx, s in enumerate(state):
        for s_idx, st in enumerate(self.state_strings):
            bins = self.bins["S_"+st]  
            found = False
            for b in range(len(bins)):
                if(state[b] >= bins[b] and state[b] <bins[b+1]):
                    norm.append(b)
                    found = True
                    break
            if(not found):
                print("Problem here!")
                norm.append(0)
        
        
        # for s_idx, s in enumerate(state):
            # found = False
            # for i in range(levels-1):
                # if(s >= i*self.step_val[s_idx]+self.min_val[s_idx] and (s < (i+1)*self.step_val[s_idx]+self.min_val[s_idx])):
                    # found = True
                    # norm.append(i)
            # if(not found):
                # norm.append(levels-1)
        # if(len(state) != len(norm)):
            # print("Problem here!")
        return norm
        

    
    def norm_actions(self, actions):
        nor =[]
        for i, s in enumerate(actions):
            nor.append( s /6.0 )
        return nor
        
    # This function should find the most similar state and action in the simulator and return the next state
    def step(self, state_sim, action_sim, init_idx):
      df_state  = self.state_similarity(self.simulator, state_sim )
      df_action = (self.action_similarity(self.simulator, action_sim))
      if(self.simulator.isnull().sum().sum()> 0):
        print("***Is there any null", self.simulator.isnull().sum().sum())
      df_both = pd.DataFrame()
      df_both["state"] = df_state
      df_both["action"] = df_action
      conditions = [
            (df_both['state'] > 0.97) & (df_both['action'] > 0.97),
            (df_both['state'] > 0.95) & (df_both['action'] > 0.95),
            (df_both['state'] > 0.90) & (df_both['action'] > 0.90),
            (df_both['state'] > 0.85) & (df_both['action'] > 0.85),
            (df_both['state'] > 0.75) & (df_both['action'] > 0.75),
            (df_both['state'] > 0.95) & (df_both['action'] > 0.65),
            (df_both['state'] > 0.65) & (df_both['action'] > 0.65),
            (df_both['state'] >= 0)   & (df_both['action'] >= 0),
            (df_both['state'] < 0),
            (df_both['action'] < 0),
            ]

        # create a list of the values we want to assign for each condition
      values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # create a new column and use np.select to assign values to it using our lists as arguments
      df_both['tier'] = np.select(conditions, values)
      
      idx = df_both['tier'].idxmin()
      idx_val =  df_both['tier'].iloc[idx]

      self.tier_hist[idx_val] +=1
      # print("df_both ", df_both)
      # print("idx_val", idx_val)  
      # print("idx", idx)  
      # print("init_idx", init_idx)  
      # print("Sim state", df_both['state'].iloc[idx])  
      # print("Sim action", df_both['action'].iloc[idx])  
      # print("Sim state init_idx ", df_both['state'].iloc[init_idx])  
      # print("Sim action init_idx ", df_both['action'].iloc[init_idx])  
      # sts = ""
      # for st in self.state_strings:
        # sts +=  str(round(self.simulator["S_"+st].iloc[idx], 2))+ " "
      # print("s found ", sts)
      # sts= ""
      # for s in state_sim:
        # sts += str(round(s, 2))+" "
      # print("c state ", sts)
      
      
      # sts = ""
      # for st in self.actions_string:
        # sts +=  str(self.simulator[st].iloc[idx])+ " "
      # print("Actions found ", sts)
      # sts = ""
      # for st in action_sim:
        # sts +=  str(st)+ " "
      # print("Curr  Actions ", sts)
      
      # sts = ""
      # for st in self.actions_string:
        # sts +=  str(self.simulator[st].iloc[init_idx])+ " "
      # print("Actions found init_idx ", sts)
      # sts = ""
      # for st in self.state_strings:
        # sts +=  str(round(self.simulator["S_"+st].iloc[init_idx], 2))+ " "
      # print("s init_idx ", sts)
      
      
      
      # input() 
      
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
      if(idx_val < 4):
          conf = True

      # print("ST==NST", state_sim == next_state)
      # print(next_state)
      # print(reward)
      # print(action_sim)
      return next_state, reward, conf
    
    def init(self):
      idx = len(self.simulator)
      idx = randint(0, idx)
      state = []
      for st in self.state_strings:
          state.append(self.simulator["S_"+st].iloc[idx])
      state = np.array(state)
      # print("---init----")
      # print("state")
      # print(state)
      # print("idx")
      # print(idx)
      return state, idx

    
    def load(self):
      self.prepare()
      # for index, row in tqdm(self.simulator.iterrows()):
          # state = []
          # next_state = []
          # actions = []
          # reward = []
          # for st in self.state_strings:
              # state.append((row["S_"+st]))
              # next_state.append((row["NS_"+st]))
          # for st in self.actions_string:
              # actions.append(row[st])
          # reward.append(row["reward"])
          # global_rewards[reward] += 1
      # print("global_rewards: ", global_rewards)              
       
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
        tot_entries = 0
        tot_high_10 = 0
        tot_high_100 = 0
        tot_high_500 = 0
        max_idx = 0
        max_val = 0
        for key in self.simulator_index:
            tot_entries += 1
            tot += self.simulator_index[key]
            if(self.simulator_index[key] > 10):
                tot_high_10 += self.simulator_index[key]
            if(self.simulator_index[key] > 100):
                tot_high_100 += self.simulator_index[key]
            if(self.simulator_index[key] > 500):
                tot_high_500 += self.simulator_index[key]
            if(self.simulator_index[key] > max_val):
                max_val = self.simulator_index[key]                  
                max_idx = key
                
        # for key in self.simulator_index:
            # if(self.simulator_index[key] > 1):
                # out += ("["+str(key)+"]:%"+str((self.simulator_index[key])/ (tot_high*1.0))+" ")
                # out += ("["+str(key)+"]:%"+str((self.simulator_index[key])/ (1))+" ")
                # out += str(key)+":"+str(self.simulator_index[key])
                # if(self.simulator_index[key]>100):
                    # print(self.simulator.loc[[self.simulator_index[key]]])
        
        
        out = " Total keys: "+ str(tot)+" tot_entries: "+str(tot_entries)
        out += " Freq: 10, 100, 500: "
        out += str(round(tot_high_10/(tot*1.0), 2))
        out += " "+ str(round(tot_high_100/(tot*1.0), 2))
        out += " "+ str(round(tot_high_500/(tot*1.0), 2))
        # out += " Hottest index: "+ str(max_idx)
        # out += " Tiers: "+np.array2string(self.tier_hist, precision=2, separator=',',
                      # suppress_small=True)
        out += " Tiers:"+' '.join([str(elem) for elem in self.tier_hist])
        return out