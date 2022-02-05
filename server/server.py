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
parser.add_argument('--lr_rate', type=float, default=1e-4, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.90, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=2, help='action scale between -1 ~ +1')
parser.add_argument("--s1", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s2", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s3", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--levels", type=int, default = 32, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.99, help = 'print interval(default : 1)')
parser.add_argument("--name", type=str, default = 'unknown')
parser.add_argument("--mlmode", type=str, default = 'training')

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
mlmode = args.mlmode

os.makedirs('./model_weights', exist_ok=True)


state_space  = 81
action_space = 4
action_scale = 3
total_reward = 0

mins = [100000000] * state_space
maxs = [0.00000001] * state_space
median=[0.04376947, 0.000939148, 0, 0, 0.573222676, 0.004596508, 0.005033913, 0.573222676, 0.036177055, 0.003903586, 0.001786599, 0.002638476, 0.002692573, 0.572481961, 0.001725929, 0.00517933, 0.572481961, 0.036754426, 0.01906055, 0.007932051, 0, 0, 0.594389009, 0.004370042, 0.001534195, 0.594389009, 0.005953031, 0.003903659, 0.003114485, 0.00268387, 0.004796958, 0.594535598, 0.000863969, 0.003060934, 0.594535598, 0.009397682, 0.030532507, 0.001989525, 0, 0, 0.594396814, 0.004063846, 0.00171645, 0.594396814, 0.01038609, 0.003903586, 0.004932507, 0, 0.004818531, 0.594014323, 0.000805707, 0.000519048, 0.594014323, 0.017739491, 0.014891342, 0, 0, 0, 0.594033408, 0.004625809, 0.002147624, 0.594033408, 0.061157953, 0.003903584, 0.004536878, 1.0643E-06, 0.005245891, 0.562137691, 0.002172376, 0.000590727, 0.562137691, 0.01732658, 0.004186702, 0.594146099, 0.002368213, 0.004026875, 0.594146099, 0.190257677, 0.046966049, 0.194642454, 0.053211177]


context_action = zmq.Context()
socket_action = context_action.socket(zmq.REP)
socket_action.bind("tcp://*:5555")

context_entery = zmq.Context()
socket_entery = context_action.socket(zmq.REP)
socket_entery.bind("tcp://*:5556")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device, levels)

agent = BQN(state_space, action_space, action_scale, learning_rate, device, s1, s2, s3, leaky)


if os.path.exists(run_name+"_out.txt"):
    os.remove(run_name+"_out.txt")
else:
    print("File does not exist")

if os.path.exists(run_name + "_states"):
    shutil.rmtree(run_name + "_states")
    os.mkdir(run_name + "_states")

def discreate_state(state):
    global mins, maxs
    for i, s in enumerate(state):
        if(s > maxs[i]):
            maxs[i] = s
        if(s < mins[i]):
            mins[i] = s
    new_state = []
    # for i, s in enumerate(state):
        # new_state.append((s-mins[i])/(maxs[i]*1.0))
    for i, s in enumerate(state):
        if(s>median[i]):
            new_state.append(1)
        else:
            new_state.append(0)
    return new_state

def action():
    print("action infinite loop")
    idx_rnd = 0
    while True:
        msg = socket_action.recv()
        state = pickle.loads(msg)
        # print("Server state", state)
        state = [float(i) for i in state]
        # state = discreate_state(state)
        action_to_send = pickle.dumps(agent.action(state, True))
        idx_rnd += 1
        if(idx_rnd == 1000):
            idx_rnd = 0
            print("Still sending an action to gem5")
        socket_action.send(action_to_send)



def get_entry():
    global total_reward
    print("train infinite loop")
    
    state_strings = [
        'core0.dtb.mmu.service_time', 'core0.dtb.mmu.wait_time', 'core0.itb.mmu.service_time', 'core0.itb.mmu.wait_time', 'core0.l1.ageTaskId', 'core0.l1.hits::total', 'core0.l1.mshrMisses::total', 'core0.l1.occupanciesTaskId', 'core0.l2.prefetcher0', 'core0.numCycles', 'core0.numSimulatedInsts', 'core0.rename.LQFullEvents', 'core0.rob.reads', 'core0.l2.ageTaskId', 'core0.l2.hits::total', 'core0.l2.mshrMisses::total', 'core0.l2.occupanciesTaskId', 'core0.l2.prefetcher0',
        
        'core1.dtb.mmu.service_time', 'core1.dtb.mmu.wait_time', 'core1.itb.mmu.service_time', 'core1.itb.mmu.wait_time', 'core1.l1.ageTaskId', 'core1.l1.hits::total', 'core1.l1.mshrMisses::total', 'core1.l1.occupanciesTaskId', 'core1.l2.prefetcher0', 'core1.numCycles', 'core1.numSimulatedInsts', 'core1.rename.LQFullEvents', 'core1.rob.reads', 'core1.l2.ageTaskId', 'core1.l2.hits::total', 'core1.l2.mshrMisses::total', 'core1.l2.occupanciesTaskId', 'core1.l2.prefetcher0',
        
        'core2.dtb.mmu.service_time', 'core2.dtb.mmu.wait_time', 'core2.itb.mmu.service_time', 'core2.itb.mmu.wait_time', 'core2.l1.ageTaskId', 'core2.l1.hits::total', 'core2.l1.mshrMisses::total', 'core2.l1.occupanciesTaskId', 'core2.l2.prefetcher0', 'core2.numCycles', 'core2.numSimulatedInsts', 'core2.rename.LQFullEvents', 'core2.rob.reads', 'core2.l2.ageTaskId', 'core2.l2.hits::total', 'core2.l2.mshrMisses::total', 'core2.l2.occupanciesTaskId', 'core2.l2.prefetcher0',
        
        'core3.dtb.mmu.service_time', 'core3.dtb.mmu.wait_time', 'core3.itb.mmu.service_time', 'core3.itb.mmu.wait_time', 'core3.l1.ageTaskId', 'core3.l1.hits::total', 'core3.l1.mshrMisses::total', 'core3.l1.occupanciesTaskId', 'core3.l2.prefetcher0', 'core3.numCycles', 'core3.numSimulatedInsts', 'core3.rename.LQFullEvents', 'core3.rob.reads', 'core3.l2.ageTaskId', 'core3.l2.hits::total', 'core3.l2.mshrMisses::total', 'core3.l2.occupanciesTaskId', 'core3.l2.prefetcher0', 
        
        'core3.mem_ctrls.avgRdBWSys', 'core3.system.l3.ageTaskId', 'core3.system.l3.hits::total', 'core3.system.l3.mshrMisses::total', 'core3.system.l3.occupanciesTaskId', 'core3.system.l3.prefetcher0', 'core3.system.l3.prefetcher1', 'core3.system.l3.prefetcher2', 'core3.system.l3.prefetcher3']
    
    actions_string = ["Core0.L1.degree", "Core0.L2.degree", "Core1.L1.degree",  "Core1.L2.degree", "Core2.L1.degree", "Core2.L2.degree", "Core3.L1.degree", "Core3.L2.degree", "LLC.P0.degree", "LLC.P1.degree", "LLC.P2.degree", "LLC.P3.degree"]
    
    # extra_info = ["S_core0_IPC", "S_core1_IPC", "S_core2_IPC", "S_core3_IPC", "NS_core0_IPC", "NS_core1_IPC", "NS_core2_IPC", "NS_core3_IPC", "total_reward"]
    extra_info = [ "total_reward"]
    
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
    with open('./csv/all_0.csv','a') as fd:  
       fd.write(lables)    
    
        
    itrs = 0
    tot_rnd = 0    
    while True:
        # conn = entry_listener.accept()
        msg = socket_entery.recv()
        entry= pickle.loads(msg)
        socket_entery.send(b"Done")
        
       
        '''
        0	core0.dtb.mmu.service_time
        1	core0.dtb.mmu.wait_time
        2	core0.itb.mmu.service_time
        3	core0.itb.mmu.wait_time
        4	core0.l1.ageTaskId
        5	core0.l1.hits::total
        6	core0.l1.mshrMisses::total
        7	core0.l1.occupanciesTaskId
        8	core0.l2.prefetcher0
        9	core0.numCycles
        10	core0.numSimulatedInsts
        11	core0.rename.LQFullEvents
        12	core0.rob.reads
        13	core0.l2.ageTaskId
        14	core0.l2.hits::total
        15	core0.l2.mshrMisses::total
        16	core0.l2.occupanciesTaskId
        17	core0.l2.prefetcher0
        18	core1.dtb.mmu.service_time
        19	core1.dtb.mmu.wait_time
        20	core1.itb.mmu.service_time
        21	core1.itb.mmu.wait_time
        22	core1.l1.ageTaskId
        23	core1.l1.hits::total
        24	core1.l1.mshrMisses::total
        25	core1.l1.occupanciesTaskId
        26	core1.l2.prefetcher0
        27	core1.numCycles
        28	core1.numSimulatedInsts
        29	core1.rename.LQFullEvents
        30	core1.rob.reads
        31	core1.l2.ageTaskId
        32	core1.l2.hits::total
        33	core1.l2.mshrMisses::total
        34	core1.l2.occupanciesTaskId
        35	core1.l2.prefetcher0
        36	core2.dtb.mmu.service_time
        37	core2.dtb.mmu.wait_time
        38	core2.itb.mmu.service_time
        39	core2.itb.mmu.wait_time
        40	core2.l1.ageTaskId
        41	core2.l1.hits::total
        42	core2.l1.mshrMisses::total
        43	core2.l1.occupanciesTaskId
        44	core2.l2.prefetcher0
        45	core2.numCycles
        46	core2.numSimulatedInsts
        47	core2.rename.LQFullEvents
        48	core2.rob.reads
        49	core2.l2.ageTaskId
        50	core2.l2.hits::total
        51	core2.l2.mshrMisses::total
        52	core2.l2.occupanciesTaskId
        53	core2.l2.prefetcher0
        54	core3.dtb.mmu.service_time
        55	core3.dtb.mmu.wait_time
        56	core3.itb.mmu.service_time
        57	core3.itb.mmu.wait_time
        58	core3.l1.ageTaskId
        59	core3.l1.hits::total
        60	core3.l1.mshrMisses::total
        61	core3.l1.occupanciesTaskId
        62	core3.l2.prefetcher0
        63	core3.numCycles
        64	core3.numSimulatedInsts
        65	core3.rename.LQFullEvents
        66	core3.rob.reads
        67	core3.l2.ageTaskId
        68	core3.l2.hits::total
        69	core3.l2.mshrMisses::total
        70	core3.l2.occupanciesTaskId
        71	core3.l2.prefetcher0
        72	core3.mem_ctrls.avgRdBWSys
        73	core3.system.l3.ageTaskId
        74	core3.system.l3.hits::total
        75	core3.system.l3.mshrMisses::total
        76	core3.system.l3.occupanciesTaskId
        77	core3.system.l3.prefetcher0
        78	core3.system.l3.prefetcher1
        79	core3.system.l3.prefetcher2
        80	core3.system.l3.prefetcher3
        '''
        # 0 entry.append(state_val_disc)
        # 1 entry.append(next_state_val)
        # 2 entry.append(new_action)
        # 3 entry.append(reward)
        # 4 entry.append(total_reward)
        # 5 entry.append(name)
        # 6 entry.append(str(sample))

                
        entry[0] = [float(i) for i in entry[0]]
        entry[1] = [float(i) for i in entry[1]]
        entry[2] = [float(i) for i in entry[2]]
        entry[3] = [float(i) for i in entry[3]]
        reward = entry[3]
        
        for idx, x in enumerate(entry[0]):
            if np.isnan(x):
                entry[0][idx] = 0
        for idx, x in enumerate(entry[1]):
            if np.isnan(x):
                entry[1][idx] = 0
        for idx, x in enumerate(entry[2]):
            if np.isnan(x):
                entry[2][idx] = 0        
        
       
                
        memory.write_buffer(entry[0], entry[1], entry[2], reward)
        
        
        with open('./csv/all_'+str(tot_rnd)+'.csv','a') as fd:
            mystring = str(entry[5])+", "+str(entry[6])+", "
            for x in entry[0]+ entry[1]+ entry[2]+ reward:
                mystring += str(x)+","
            mystring+= str(entry[4])+"\n"
            fd.write(mystring)
     
        itrs += 1
        if(itrs == 100*1000):
            itrs = 0
            tot_rnd += 1
        
        
    



def train():
    print("train infinite loop ", batch_size)
    loss_itr = 0
    while True:
        if(memory.size() > batch_size):
            loss = agent.train_model(memory, batch_size, gamma)
            loss_itr += 1
            # 
            if(loss_itr == 5000):
                print("Loss:", loss.item())
                agent.save_model("model")
                loss_itr = 0


if __name__ == "__main__":
    # creating thread
    t1 = threading.Thread(target=action, args=())
    t2 = threading.Thread(target=get_entry, args=())
    t3 = threading.Thread(target=train, args=())

    t1.start()
    t2.start()
    if(mlmode == "train"):
        print("-----train")
        t3.start()

    t1.join()
    t2.join()
    if(mlmode == "train"):
        print("-----train")
        t3.join()

    # both threads completely executed
    print("Done!")
