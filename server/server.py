from multiprocessing.connection import Listener
import numpy as np
import threading
import time
import pickle
from utils import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import argparse
from tqdm import tqdm
from agent import BQN
import torch.autograd.profiler as profiler
import shutil
from random import random
from random import randint
import random
import zmq




parser = argparse.ArgumentParser('parameters')
parser.add_argument('--lr_rate', type=float, default=1e-2, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=2, help='action scale between -1 ~ +1')
parser.add_argument("--s1", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s2", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s3", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--levels", type=int, default = 32, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.95, help = 'print interval(default : 1)')
parser.add_argument("--name", type=str, default = 'unknown')

args = parser.parse_args()

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


os.makedirs('./model_weights', exist_ok=True)


state_space  = 65
action_space = 19
action_scale = 2
total_reward = 0

# action_address = ('localhost', 6000)
# action_listener = Listener(action_address, authkey=b'secret password')

# entry_address = ('localhost', 7000)
# entry_listener = Listener(entry_address, authkey=b'secret password')

context_action = zmq.Context()
socket_action = context_action.socket(zmq.REP)
socket_action.bind("tcp://*:5555")

context_entery = zmq.Context()
socket_entery = context_action.socket(zmq.REP)
socket_entery.bind("tcp://*:5556")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(100000, action_space, device, levels)

agent = BQN(state_space, action_space, action_scale, learning_rate, device, s1, s2, s3, leaky)


if os.path.exists(run_name+"_out.txt"):
    os.remove(run_name+"_out.txt")
else:
    print("File does not exist")

if os.path.exists(run_name + "_states"):
    shutil.rmtree(run_name + "_states")
    os.mkdir(run_name + "_states")

def discreate_state(state):
    
    mins= [59, 75335, 142192, 500000, 0, 0, 630020, 0.119366847, 0, 0, 0.12049345, 0, 0, 0, 0, 43921, 142192, 8505, 0, 0, 142903, 0.039334033, 0, 0, 0.039430155, 0, 0, 0, 9, 33954, 142098, 9912, 0, 0, 173352, 0.025957558, 0, 0, 0.027673458, 0, 0, 0, 5, 28853, 142192, 7421, 0, 0, 101384, 0.033197159, 0, 0, 0.035413655, 0, 0, 0, 0, 16, 0, 0, 2, 2, 0, 0, 0]
    maxs= [5714209, 5721438, 6841388, 500031, 2846625, 2854303, 7332256, 3.763425509, 0.13899086, 214138, 3.822542759, 3753, 42983, 233196, 5639302, 5796162, 6841387, 5780050, 2406223, 2445696, 9558429, 2.744094844, 0.223608911, 910516, 2.753039322, 11288, 172432, 501478, 5045654, 5030090, 6841387, 8992712, 1135622, 2347751, 16802317, 3.109019221, 0.664371208, 463988, 3.109961962, 7428, 163375, 487453, 6442789, 6509941, 6841387, 9185860, 1397413, 3081093, 14723068, 3.105343698, 0.664287675, 498729, 3.106354828, 6235, 96388, 426338, 165984, 283254, 13191, 1867702606, 261944, 280087, 354590, 2067986, 239437]
    medians= [37647.5, 162229.5, 397825, 500009, 763, 4034.5, 1026473, 1.381291635, 0.006053663, 224.5, 1.397617067, 0, 642, 4831, 94593.5, 208773.5, 397828, 511957.5, 2960, 13995, 1004330, 1.317464247, 0.007398986, 1283, 1.329838984, 0, 557, 5475, 109457.5, 222522.5, 397834, 428393.5, 1297, 17050, 923284.5, 1.2092372, 0.007497391, 3188.5, 1.222988998, 0, 670, 7914, 107351, 232894, 397830, 455646, 2647, 15924, 954413, 1.280733131, 0.007309875, 2324.5, 1.29517581, 0, 768, 6737, 3194.5, 11548, 266, 109998128, 10680, 12265, 5297, 25260, 7254]



    new_state = []
    for i, s in enumerate(state):
        # if(s> medians[i]):
            # new_state.append(1)
        # else:
            # new_state.append(0)
        new_state.append((s-mins[i])/(maxs[i]*1.0))
    return new_state

def action():
    print("action infinite loop")
    while True:
        msg = socket_action.recv()
        state = pickle.loads(msg)
        state = [float(i) for i in state]
        state = discreate_state(state)
        action_to_send = pickle.dumps(agent.action(state, True))
        print("sending an action to gem5")
        socket_action.send(action_to_send)



def get_entry():
    global total_reward
    print("train infinite loop")
    state_strings = ["core0.decode.blockedCycles", "core0.fetch.cycles", "core0.numCycles", "core0.numSimulatedInsts", "core0.rename.LQFullEvents", "core0.rename.unblockCycles", "core0.rob.reads", "core0.switch_cpus0.numRate", "core0.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core0.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core0.system.switch_cpus0.issueRate", "core0.timesIdled", "core0.l2cache.ReadReq.hits::total", "core0.l2cache.demandAccesses::total", "core1.decode.blockedCycles", "core1.fetch.cycles", "core1.numCycles", "core1.numSimulatedInsts", "core1.rename.LQFullEvents", "core1.rename.unblockCycles", "core1.rob.reads", "core1.switch_cpus0.numRate", "core1.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core1.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core1.system.switch_cpus0.issueRate", "core1.timesIdled", "core1.l2cache.ReadReq.hits::total", "core1.l2cache.demandAccesses::total", "core2.decode.blockedCycles", "core2.fetch.cycles", "core2.numCycles", "core2.numSimulatedInsts", "core2.rename.LQFullEvents", "core2.rename.unblockCycles", "core2.rob.reads", "core2.switch_cpus0.numRate", "core2.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core2.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core2.system.switch_cpus0.issueRate", "core2.timesIdled", "core2.l2cache.ReadReq.hits::total", "core2.l2cache.demandAccesses::total", "core3.decode.blockedCycles", "core3.fetch.cycles", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents", "core3.rename.unblockCycles", "core3.rob.reads", "core3.switch_cpus0.numRate", "core3.system.cpu0.dcache.ReadReq.mshrMissRate::total", "core3.system.cpu0.dcache.prefetcher.prefetchers1.pfIssued", "core3.system.switch_cpus0.issueRate", "core3.timesIdled", "core3.l2cache.ReadReq.hits::total", "core3.l2cache.demandAccesses::total", "core3.ReadSharedReq.mshrMisses::total", "core3.mem_ctrls.numStayReadState", "core3.mem_ctrls.rdQLenPdf::3", "core3.mem_ctrls.totGap", "core3.system.l3.ReadSharedReq.accesses::total", "core3.system.l3.demandAccesses::total", "core3.system.l3.prefetcher.prefetchersx.pfSpanPage", "core3.system.l3.tags.totalRefs", "core3.system.mem_ctrls.requestorReadAccesses::cpu0.dcache.prefetcher.prefetchers1"]
    actions_string = ["Core0.L1.P0.degree", "Core0.L1.P1.degree", "Core0.L2.P0.degree", "Core0.L2.P1.degree", "Core1.L1.P0.degree", "Core1.L1.P1.degree", "Core1.L2.P0.degree", "Core1.L2.P1.degree", "Core2.L1.P0.degree", "Core2.L1.P1.degree", "Core2.L2.P0.degree", "Core2.L2.P1.degree", "Core3.L1.P0.degree", "Core3.L1.P1.degree", "Core3.L2.P0.degree", "Core3.L2.P1.degree" , "LLC.P1.degree", "LLC.P2.degree", "LLC.P0.degree"]
    
    extra_info = ["S_core0_IPC", "S_core1_IPC", "S_core2_IPC", "S_core3_IPC", "NS_core0_IPC", "NS_core1_IPC", "NS_core2_IPC", "NS_core3_IPC"]
    
    filePath = "./all.csv"
    if os.path.exists(filePath):
        os.remove(filePath)
    else:
        print("Can not delete the file as it doesn't exists")
    
    lables = "apps, samples, "
    for s in state_strings :
        lables += "S_"+s+","
    for s in state_strings:
        lables += "NS_"+s+","
    for s in actions_string :
       lables += s+","
    lables += "reward,"
    
    for s in extra_info :
       lables += s+","
    lables += "\n"
    with open('all.csv','a') as fd:  
       fd.write(lables)    
       
    file1 = open("reward.txt", "w")
    file1.close()
    
    
    itrs = 0
        
    while True:
        # conn = entry_listener.accept()
        msg = socket_entery.recv()
        entry= pickle.loads(msg)
        socket_entery.send(b"Done")
        entry[0] = [float(i) for i in entry[0]]
        entry[1] = [float(i) for i in entry[1]]
        entry[2] = [float(i) for i in entry[2]]
        
        for idx, x in enumerate(entry[0]):
            if np.isnan(x):
                entry[0][idx] = 0
        for idx, x in enumerate(entry[1]):
            if np.isnan(x):
                entry[1][idx] = 0
        for idx, x in enumerate(entry[2]):
            if np.isnan(x):
                entry[2][idx] = 0        
        
        reward = [0] 
        S_core0_IPC = (float(entry[0][3]/(0.000001+entry[0][2])))
        S_core1_IPC = (float(entry[0][17]/(0.000001+entry[0][16])))
        S_core2_IPC = (float(entry[0][31]/(0.000001+entry[0][30])))
        S_core3_IPC = (float(entry[0][45]/(0.000001+entry[0][44])))
        S_core_IPC = (S_core0_IPC+S_core1_IPC+S_core2_IPC+S_core3_IPC)/4
        
        NS_core0_IPC = (float(entry[1][3]/(0.000001+entry[1][2])))
        NS_core1_IPC = (float(entry[1][17]/(0.000001+entry[1][16])))
        NS_core2_IPC = (float(entry[1][31]/(0.000001+entry[1][30])))
        NS_core3_IPC = (float(entry[1][45]/(0.000001+entry[1][44])))
        
        # NS_core_IPC = (NS_core0_IPC+NS_core1_IPC+NS_core2_IPC+NS_core3_IPC)/4
        
        diff = ((NS_core0_IPC/S_core0_IPC)-1)+ ((NS_core1_IPC/S_core1_IPC)-1)+ ((NS_core2_IPC/S_core2_IPC)-1)+ ((NS_core3_IPC/S_core3_IPC)-1)
        if not np.isnan(diff):
            reward[0] = int(diff*100)
            # if(diff > 0):
                # reward[0] = 1
            # else:
                # reward[0] = -1
                
        total_reward += reward[0]
        print(str(entry[3])+" reward", reward, total_reward, memory.size(), S_core0_IPC, S_core1_IPC, S_core2_IPC, S_core3_IPC, " --- ", NS_core0_IPC, NS_core1_IPC, NS_core2_IPC, NS_core3_IPC)
        file1 = open("reward.txt", "a")
        st = str(entry[3])+" reward:"+str(reward)+" total_reward:"+str(total_reward)
        file1.write(st+"\n")
        file1.close()
        # print(type(reward), type(reward[0]))
        memory.write_buffer(discreate_state(entry[0]), discreate_state(entry[1]), entry[2], reward)
        entry[0] = discreate_state(entry[0])
        entry[1] = discreate_state(entry[1])

        with open('all.csv','a') as fd:
            mystring = str(entry[3])+", "+str(entry[4])+", "
            for x in entry[0]+ entry[1]+ entry[2]+ reward:
                mystring += str(x)+","
            
            mystring+= str(S_core0_IPC)+","
            mystring+= str(S_core1_IPC)+","
            mystring+= str(S_core2_IPC)+","
            mystring+= str(S_core3_IPC)+","
            
            mystring+= str(NS_core0_IPC)+","
            mystring+= str(NS_core1_IPC)+","
            mystring+= str(NS_core2_IPC)+","
            mystring+= str(NS_core3_IPC)+"\n"
            
 
            fd.write(mystring)
        itrs  += 1
        if(itrs == 120*200):
            total_reward = 0
        # memory.print_buffer()
        
    # entry_listener.close()


def train():
    print("train infinite loop")
    loss_itr = 0
    while True:
        if(memory.size() > batch_size):
            loss = agent.train_model(memory, batch_size, gamma)
            loss_itr += 1
            
            if(loss_itr%100000 == 0):
                agent.save_model("model_"+str(loss_itr))
                print("Loss:", loss.item())
                loss_itr = 0


if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=action, args=())
    t2 = threading.Thread(target=get_entry, args=())
    t3 = threading.Thread(target=train, args=())

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    # both threads completely executed
    print("Done!")
