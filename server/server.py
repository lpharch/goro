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
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument("--levels", type=int, default = 32, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.99, help = 'print interval(default : 1)')
parser.add_argument("--name", type=str, default = 'unknown')
parser.add_argument("--mlmode", type=str, default = 'training')

args = parser.parse_args()

learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
run_name = args.name
leaky = args.leaky
levels = args.levels
mlmode = args.mlmode

os.makedirs('./model_weights', exist_ok=True)


state_space  = 28
action_space = 20
action_scale = 2
total_reward = 0

mins = [100000000] * state_space
maxs = [0.00000001] * state_space
median=[]
'''
    * 0	core0.dtb.mmu.service_time
    1	core0.dtb.mmu.wait_time
    2	core0.itb.mmu.service_time
    3	core0.itb.mmu.wait_time
    4	core0.l1.ageTaskId
    * 5	core0.l1.hits::total
    * 6	core0.l1.mshrMisses::total
    7	core0.l1.occupanciesTaskId
    8	core0.l2.prefetcher0
    9	core0.l2.prefetcher1
    10	core0.numCycles
    11	core0.numSimulatedInsts
    12	core0.rename.LQFullEvents
    * 13	core0.rob.reads
    14	core0.l2.ageTaskId
    * 15	core0.l2.hits::total
    * 16	core0.l2.mshrMisses::total
    17	core0.l2.occupanciesTaskId
    18	core0.l2.prefetcher0
    19	core0.l2.prefetcher1
    
    20	core1.dtb.mmu.service_time
    21	core1.dtb.mmu.wait_time
    22	core1.itb.mmu.service_time
    23	core1.itb.mmu.wait_time
    24	core1.l1.ageTaskId
    25	core1.l1.hits::total
    26	core1.l1.mshrMisses::total
    27	core1.l1.occupanciesTaskId
    28	core1.l2.prefetcher0
    29	core1.l2.prefetcher1
    30	core1.numCycles
    31	core1.numSimulatedInsts
    32	core1.rename.LQFullEvents
    33	core1.rob.reads
    34	core1.l2.ageTaskId
    35	core1.l2.hits::total
    36	core1.l2.mshrMisses::total
    37	core1.l2.occupanciesTaskId
    38	core1.l2.prefetcher0
    39	core1.l2.prefetcher1
    40	core2.dtb.mmu.service_time
    41	core2.dtb.mmu.wait_time
    42	core2.itb.mmu.service_time
    43	core2.itb.mmu.wait_time
    44	core2.l1.ageTaskId
    45	core2.l1.hits::total
    46	core2.l1.mshrMisses::total
    47	core2.l1.occupanciesTaskId
    48	core2.l2.prefetcher0
    49	core2.l2.prefetcher1
    50	core2.numCycles
    51	core2.numSimulatedInsts
    52	core2.rename.LQFullEvents
    53	core2.rob.reads
    54	core2.l2.ageTaskId
    55	core2.l2.hits::total
    56	core2.l2.mshrMisses::total
    57	core2.l2.occupanciesTaskId
    58	core2.l2.prefetcher0
    59	core2.l2.prefetcher1
    60	core3.dtb.mmu.service_time
    61	core3.dtb.mmu.wait_time
    62	core3.itb.mmu.service_time
    63	core3.itb.mmu.wait_time
    64	core3.l1.ageTaskId
    65	core3.l1.hits::total
    66	core3.l1.mshrMisses::total
    67	core3.l1.occupanciesTaskId
    68	core3.l2.prefetcher0
    69	core3.l2.prefetcher1
    70	core3.numCycles
    71	core3.numSimulatedInsts
    72	core3.rename.LQFullEvents
    73	core3.rob.reads
    74	core3.l2.ageTaskId
    75	core3.l2.hits::total
    76	core3.l2.mshrMisses::total
    77	core3.l2.occupanciesTaskId
    78	core3.l2.prefetcher0
    79	core3.l2.prefetcher1
    
    * 80	core3.mem_ctrls.avgRdBWSys
    * 81	core3.system.l3.ageTaskId
    * 82	core3.system.l3.hits::total
    * 83	core3.system.l3.mshrMisses::total
    84	core3.system.l3.occupanciesTaskId
    85	core3.system.l3.prefetcher0
    86	core3.system.l3.prefetcher1
    87	core3.system.l3.prefetcher2

'''
needed_index = [ 0,  5,  6, 13, 15, 16,  
                20, 25, 26, 33, 35, 36,  
                40, 45, 46, 53, 55, 56, 
                60, 65, 66, 73, 75, 76,
                80, 81, 82, 83
               ]
                       
                       
context_action = zmq.Context()
socket_action = context_action.socket(zmq.REP)
socket_action.bind("tcp://*:5555")

context_entery = zmq.Context()
socket_entery = context_action.socket(zmq.REP)
socket_entery.bind("tcp://*:5556")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device, levels)

agent = BQN(state_space, action_space, action_scale, learning_rate, device)


if os.path.exists(run_name+"_out.txt"):
    os.remove(run_name+"_out.txt")
else:
    print("File does not exist")

if os.path.exists(run_name + "_states"):
    shutil.rmtree(run_name + "_states")
    os.mkdir(run_name + "_states")


def action():
    print("action infinite loop")
    idx_rnd = 0
    while True:
        msg = socket_action.recv()
        state_all = pickle.loads(msg)
        state = []
        for i in needed_index:
            state.append( float(state_all[i]))
        
        action_to_send = pickle.dumps(agent.action(state, idx_rnd))
        idx_rnd += 1
        if(idx_rnd == 1000):
            idx_rnd = 0
            print("Still sending an action to gem5")
        socket_action.send(action_to_send)



def get_entry():
    global total_reward
    print("train infinite loop")
    
    filePath = "./all.csv"
    if os.path.exists(filePath):
        os.remove(filePath)
    else:
        print("Can not delete the file as it doesn't exists")
    
    lables = "apps, samples, "

    
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
            
              entry.append(state_val_disc)
              entry.append(next_state_val)
              entry.append(new_action)        
              entry.append(reward)entry[0] = [float(i) for i in entry[0]]
              entry.append(total_reward)entry[1] = [float(i) for i in entry[1]]
              entry.append(name)entry[2] = [float(i) for i in entry[2]]
              entry.append(str(sample))entry[3] = [float(i) for i in entry[3]]
            
        '''

        state = []
        next_state = []
        
        for i in needed_index:
            state.append( float(entry[0][i]))
        for i in needed_index:
            next_state.append( float(entry[1][i]))        
        
        
        entry[2] = [float(i) for i in entry[2]]
        entry[3] = [float(i) for i in entry[3]]
        reward = entry[3]
        
        for idx, x in enumerate(state):
            if np.isnan(x):
                state[idx] = 0
        for idx, x in enumerate(next_state):
            if np.isnan(x):
                next_state[idx] = 0
        for idx, x in enumerate(entry[2]):
            if np.isnan(x):
                entry[2][idx] = 0        
        
       
                
        memory.write_buffer(state, next_state, entry[2], reward)
        
        
        with open('./csv/all_'+str(tot_rnd)+'.csv','a') as fd:
            mystring = str(entry[5])+", "+str(entry[6])+", "
            for x in state+ next_state+ entry[2]+ reward:
                mystring += str(x)+","
            mystring+= str(entry[4])+"\n"
            fd.write(mystring)
     
        itrs += 1
        if(itrs == 10*1000):
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
