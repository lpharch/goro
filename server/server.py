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
parser.add_argument('--batch_size', type=int, default=128, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
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


state_space  = 88
action_space = 19
action_scale = 2
total_reward = 0

mins = [100000000] * state_space
maxs = [0.00000001] * state_space

print("mins ", mins )
print("maxs ", maxs )


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
memory = ReplayBuffer(5000, action_space, device, levels)

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
    for i, s in enumerate(state):
        new_state.append((s-mins[i])/(maxs[i]*1.0))
    return new_state

def action():
    print("action infinite loop")
    idx_rnd = 0
    while True:
        msg = socket_action.recv()
        state = pickle.loads(msg)
        state = [float(i) for i in state]
        state = discreate_state(state)
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
        "core0.dtb.mmu.service_time", "core0.dtb.mmu.wait_time", "core0.itb.mmu.service_time", "core0.itb.mmu.wait_time", "core0.l1.ageTaskId", "core0.l1.hits::total", "core0.l1.mshrMisses::total", "core0.l1.occupanciesTaskId", "core0.l2.prefetcher0", "core0.l2.prefetcher1", "core0.numCycles", "core0.numSimulatedInsts", "core0.rename.LQFullEvents", "core0.rob.reads", "core0.l2.ageTaskId", "core0.l2.hits::total", "core0.l2.mshrMisses::total", "core0.l2.occupanciesTaskId", "core0.l2.prefetcher0", "core0.l2.prefetcher1",

        "core1.dtb.mmu.service_time", "core1.dtb.mmu.wait_time", "core1.itb.mmu.service_time", "core1.itb.mmu.wait_time", "core1.l1.ageTaskId", "core1.l1.hits::total", "core1.l1.mshrMisses::total", "core1.l1.occupanciesTaskId", "core1.l2.prefetcher0", "core1.l2.prefetcher1", "core1.numCycles", "core1.numSimulatedInsts", "core1.rename.LQFullEvents", "core1.rob.reads", "core1.l2.ageTaskId", "core1.l2.hits::total", "core1.l2.mshrMisses::total", "core1.l2.occupanciesTaskId", "core1.l2.prefetcher0", "core1.l2.prefetcher1", 
        
        "core2.dtb.mmu.service_time", "core2.dtb.mmu.wait_time", "core2.itb.mmu.service_time", "core2.itb.mmu.wait_time", "core2.l1.ageTaskId", "core2.l1.hits::total", "core2.l1.mshrMisses::total", "core2.l1.occupanciesTaskId", "core2.l2.prefetcher0", "core2.l2.prefetcher1", "core2.numCycles", "core2.numSimulatedInsts", "core2.rename.LQFullEvents", "core2.rob.reads", "core2.l2.ageTaskId", "core2.l2.hits::total", "core2.l2.mshrMisses::total", "core2.l2.occupanciesTaskId", "core2.l2.prefetcher0", "core2.l2.prefetcher1",

        "core3.dtb.mmu.service_time", "core3.dtb.mmu.wait_time", "core3.itb.mmu.service_time", "core3.itb.mmu.wait_time", "core3.l1.ageTaskId", "core3.l1.hits::total", "core3.l1.mshrMisses::total", "core3.l1.occupanciesTaskId", "core3.l2.prefetcher0", "core3.l2.prefetcher1", "core3.numCycles", "core3.numSimulatedInsts", "core3.rename.LQFullEvents", "core3.rob.reads", "core3.l2.ageTaskId", "core3.l2.hits::total", "core3.l2.mshrMisses::total", "core3.l2.occupanciesTaskId", "core3.l2.prefetcher0", "core3.l2.prefetcher1", 
        
        "core3.mem_ctrls.avgRdBWSys", 
        
        "core3.system.l3.ageTaskId", "core3.system.l3.hits::total", "core3.system.l3.mshrMisses::total", "core3.system.l3.occupanciesTaskId", "core3.system.l3.prefetcher0", "core3.system.l3.prefetcher1", "core3.system.l3.prefetcher2"]
    
    actions_string = ["Core0.L1.P0.degree", "Core0.L1.P1.degree", "Core0.L2.P0.degree", "Core0.L2.P1.degree", "Core1.L1.P0.degree", "Core1.L1.P1.degree", "Core1.L2.P0.degree", "Core1.L2.P1.degree", "Core2.L1.P0.degree", "Core2.L1.P1.degree", "Core2.L2.P0.degree", "Core2.L2.P1.degree", "Core3.L1.P0.degree", "Core3.L1.P1.degree", "Core3.L2.P0.degree", "Core3.L2.P1.degree" , "LLC.P1.degree", "LLC.P2.degree", "LLC.P0.degree"]
    
    extra_info = ["S_core0_IPC", "S_core1_IPC", "S_core2_IPC", "S_core3_IPC", "NS_core0_IPC", "NS_core1_IPC", "NS_core2_IPC", "NS_core3_IPC", "total_reward"]
    
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
        
        # 0 entry.append(state_val)
        # 1 entry.append(next_state_val)
        # 2 entry.append(new_action)
        # 3 entry.append(name)
        # 4 entry.append(str(sample))

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
        9	core0.l2.prefetcher1
        10	core0.numCycles
        11	core0.numSimulatedInsts
        12	core0.rename.LQFullEvents
        13	core0.rob.reads
        14	core0.l2.ageTaskId
        15	core0.l2.hits::total
        16	core0.l2.mshrMisses::total
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
        80	core3.mem_ctrls.avgRdBWSys
        81	core3.system.l3.ageTaskId
        82	core3.system.l3.hits::total
        83	core3.system.l3.mshrMisses::total
        84	core3.system.l3.occupanciesTaskId
        85	core3.system.l3.prefetcher0
        86	core3.system.l3.prefetcher1
        87	core3.system.l3.prefetcher2
        '''
        
                
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
        S_core0_IPC = entry[0][11]/entry[0][10]*1.0
        S_core1_IPC = entry[0][31]/entry[0][30]*1.0
        S_core2_IPC = entry[0][51]/entry[0][50]*1.0
        S_core3_IPC = entry[0][71]/entry[0][70]*1.0
        
        NS_core0_IPC =  entry[1][11]/entry[1][10]*1.0
        NS_core1_IPC =  entry[1][31]/entry[1][30]*1.0
        NS_core2_IPC =  entry[1][51]/entry[1][50]*1.0
        NS_core3_IPC =  entry[1][71]/entry[1][70]*1.0
        

        
        diff = ((NS_core0_IPC/S_core0_IPC)-1)+ ((NS_core1_IPC/S_core1_IPC)-1)+ ((NS_core2_IPC/S_core2_IPC)-1)+ ((NS_core3_IPC/S_core3_IPC)-1)
        if not np.isnan(diff):
            reward[0] = int(diff*100)
                
        total_reward += reward[0]

        memory.write_buffer(discreate_state(entry[0]), discreate_state(entry[1]), entry[2], reward)
        entry[0] = discreate_state(entry[0])
        entry[1] = discreate_state(entry[1])
        # print("here")
        with open('./csv/all_'+str(tot_rnd)+'.csv','a') as fd:
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
            mystring+= str(NS_core3_IPC)+","
            mystring+= str(total_reward)+"\n"
            
            # print("mystring", mystring)
            fd.write(mystring)
        
        
        # entry[5].to_csv('my_csv.csv', mode='a')
        
        itrs += 1
        if(itrs == 100*1000):
            itrs = 0
            tot_rnd += 1
            total_reward = 0
        # memory.print_buffer()
    
# entry_listener.close()


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
