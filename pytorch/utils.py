import torch
import pandas as pd
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
    
    def reward_gen(self, num):
        if(num >= 4):
            return int(num-4)
        else:
            return -1
        
    def print_buffer(self):
        print(self.buffer)
    
    def load(self, path):
      df = pd.read_csv(path)
      states_trings = {"core0.decode.blockedCycles", "core0.fetch.cycles", "core0.numCycles", "core0.numSimulatedInsts", "core0.rename.LQFullEvents", "core0.rename.unblockCycles", "core0.rob.reads", "core0.switch_cpus0.numRate", "core0.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core0.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core0.system.switch_cpus0.issueRate", "core0.timesIdled", "core0.l2cache.ReadReq.hits::total", "core0.l2cache.demandAccesses::total", "core1.decode.blockedCycles", "core1.fetch.cycles", "core1.numCycles", "core1.numSimulatedInsts", "core1.rename.LQFullEvents", "core1.rename.unblockCycles", "core1.rob.reads", "core1.switch_cpus0.numRate", "core1.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core1.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core1.system.switch_cpus0.issueRate", "core1.timesIdled", "core1.l2cache.ReadReq.hits::total", "core1.l2cache.demandAccesses::total", "core2.decode.blockedCycles", "core2.fetch.cycles", "core2.numCycles", "core2.numSimulatedInsts", "core2.rename.LQFullEvents", "core2.rename.unblockCycles", "core2.rob.reads", "core2.switch_cpus0.numRate", "core2.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core2.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core2.system.switch_cpus0.issueRate", "core2.timesIdled", "core2.l2cache.ReadReq.hits::total", "core2.l2cache.demandAccesses::total", "core3.decode.blockedCycles", "core3.fetch.cycles", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents", "core3.rename.unblockCycles", "core3.rob.reads", "core3.switch_cpus0.numRate", "core3.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core3.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core3.system.switch_cpus0.issueRate", "core3.timesIdled", "core3.l2cache.ReadReq.hits::total", "core3.l2cache.demandAccesses::total", "core3.ReadSharedReq.mshrMisses::total", "core3.mem_ctrls.numStayReadState", "core3.mem_ctrls.rdQLenPdf::3", "core3.mem_ctrls.totGap", "core3.system.l3.ReadSharedReq.accesses::total", "core3.system.l3.demandAccesses::total", "core3.system.l3.prefetcher.prefetchersx.pfSpanPage", "core3.system.l3.tags.totalRefs", "core3.system.mem_ctrls.requestorReadAccesses::cpu0.dcache.prefetcher.prefetchers1"} 
      # 
      
      actions_string = {"Core0.L1.P0.degree", "Core0.L1.P1.degree", "Core0.L2.P0.degree", "Core0.L2.P1.degree", "Core1.L1.P0.degree", "Core1.L1.P1.degree", "Core1.L2.P0.degree", "Core1.L2.P1.degree", "Core2.L1.P0.degree", "Core2.L1.P1.degree", "Core2.L2.P0.degree", "Core2.L2.P1.degree", "Core3.L1.P0.degree", "Core3.L1.P1.degree", "Core3.L2.P0.degree", "Core3.L2.P1.degree" , "LLC.P1.degree", "LLC.P2.degree", "LLC.P0.degree"}
      #
           
      df["reward"] = (df["S_core0.numSimulatedInsts"]+df["S_core1.numSimulatedInsts"]+df["S_core2.numSimulatedInsts"]+df["S_core3.numSimulatedInsts"])/(df["S_core0.numCycles"]+df["S_core1.numCycles"]+df["S_core2.numCycles"]+df["S_core3.numCycles"])
      
      df["reward"] = df["reward"].apply(self.reward_gen)
      for index, row in df.iterrows():
          state = []
          next_state = []
          actions = []
          reward = []
          for st in states_trings:
              state.append((row["S_"+st]))
              next_state.append((row["NS_"+st]))
          for st in actions_string:
              actions.append(row[st])
          reward.append(row["reward"])
          
          
          state = np.array(state)
          next_state = np.array(next_state)
          actions = np.array(actions)
          reward = np.array(reward)[0]
 
          # print("State")  
          # print(type(state))
          # print(len(state))
          # print("next_state")  
          # print(next_state)
          # print("actions")  
          # print(type(actions))
          # print("reward")  
          # print(reward)
          # print("----")
          self.put((state, actions, reward, next_state, 0))
          global_rewards[reward] += 1
                    
       
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst, actions_lst = [], [], [], [], []
        # print("**********self.action_space", self.action_space)
        actions_lst = [[] for i in range(self.action_space)]
        
        
        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            # actions_lst.append(actions)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(self.device) for x in actions_lst]
        # print("actions_lst", actions_lst)
        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
               actions_lst ,\
               torch.tensor(reward_lst).to(self.device),\
               torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
               torch.tensor(done_mask_lst).to(self.device)
    def size(self):
        return len(self.buffer)
        
    def info(self):
        print("Reward info ", global_rewards)