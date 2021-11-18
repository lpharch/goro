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
parser.add_argument('--action_scale', type=int, default=2, help='action scale between -1 ~ +1')
parser.add_argument("--s1", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s2", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--s3", type=int, default = 1, help = 'print interval(default : 1)')
parser.add_argument("--levels", type=int, default = 32, help = 'print interval(default : 1)')
parser.add_argument("--leaky", type=float, default = 0.99, help = 'print interval(default : 1)')
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


state_space  = 31
action_space = 19
action_scale = 2
total_reward = 0

mins = [100000000] * state_space
maxs = [0] * state_space

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
memory = ReplayBuffer(2000, action_space, device, levels)

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
    state_strings = ["core0.IPC", "core0.LQFullEvents", "core0.rob.reads", "core0.system.l1.hits::total", "core0.system.l1.mshrMisses::total", "core0.system.l2.hits::total", "core0.system.l2.mshrMisses::total", "core1.IPC", "core1.LQFullEvents", "core0.rob.reads", "core1.system.l1.hits::total", "core1.system.l1.mshrMisses::total", "core0.system.l2.hits::total", "core1.system.l2.mshrMisses::total", "core2.IPC", "core2.LQFullEvents", "core2.rob.reads", "core2.system.l1.hits::total", "core2.system.l1.mshrMisses::total", "core2.system.l2.hits::total", "core2.system.l2.mshrMisses::total", "core3.IPC", "core3.LQFullEvents", "core3.rob.reads", "core3.system.l1.hits::total", "core3.system.l1.mshrMisses::total", "core3.system.l2.hits::total", "core3.system.l2.mshrMisses::total", "mem_ctrls.avgRdBWSys", "core3.system.l3.hits::total", "system.l3.mshrMisses::total"]
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
    
       
    # file1 = open("reward.txt", "w")
    # file1.close()
    
    
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
        # 5 entry.append(next_state_all)
        
        # S_core0.IPC	0
        # S_core0.LQFullEvents	1
        # S_core0.rob.reads	2
        # S_core0.system.l1.hits::total	3
        # S_core0.system.l1.mshrMisses::total	4
        # S_core0.system.l2.hits::total	5
        # S_core0.system.l2.mshrMisses::total	6
        # S_core1.IPC	7
        # S_core1.LQFullEvents	8
        # S_core0.rob.reads	9
        # S_core1.system.l1.hits::total	10
        # S_core1.system.l1.mshrMisses::total	11
        # S_core0.system.l2.hits::total	12
        # S_core1.system.l2.mshrMisses::total	13
        # S_core2.IPC	14
        # S_core2.LQFullEvents	15
        # S_core2.rob.reads	16
        # S_core2.system.l1.hits::total	17
        # S_core2.system.l1.mshrMisses::total	18
        # S_core2.system.l2.hits::total	19
        # S_core2.system.l2.mshrMisses::total	20
        # S_core3.IPC	21
        # S_core3.LQFullEvents	22
        # S_core3.rob.reads	23
        # S_core3.system.l1.hits::total	24
        # S_core3.system.l1.mshrMisses::total	25
        # S_core3.system.l2.hits::total	26
        # S_core3.system.l2.mshrMisses::total	27
        # S_mem_ctrls.avgRdBWSys	28
        # S_core3.system.l3.hits::total	29
        # S_system.l3.mshrMisses::total	30
    
            
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
        S_core0_IPC = entry[0][0]
        S_core1_IPC = entry[0][7]
        S_core2_IPC = entry[0][14]
        S_core3_IPC = entry[0][21]
        
        NS_core0_IPC =  entry[1][0]
        NS_core1_IPC =  entry[1][7]
        NS_core2_IPC =  entry[1][14]
        NS_core3_IPC =  entry[1][21]
        

        
        # diff = ((NS_core0_IPC/S_core0_IPC)-1)+ ((NS_core1_IPC/S_core1_IPC)-1)+ ((NS_core2_IPC/S_core2_IPC)-1)+ ((NS_core3_IPC/S_core3_IPC)-1)
        diff = ((NS_core3_IPC/S_core3_IPC)-1)
        if not np.isnan(diff):
            reward[0] = int(diff*100)
            # if(diff > 0):
                # reward[0] = 1
            # else:
                # reward[0] = -1
                
        total_reward += reward[0]
        # print(str(entry[3])+" reward", reward[0], total_reward, memory.size(), S_core0_IPC, S_core1_IPC, S_core2_IPC, S_core3_IPC, " --- ", NS_core0_IPC, NS_core1_IPC, NS_core2_IPC, NS_core3_IPC)
        # file1 = open("reward.txt", "a")
        # st = str(entry[3])+" reward:"+str(reward)+" total_reward:"+str(total_reward)
        # file1.write(st+"\n")
        # file1.close()
        # print(type(reward), type(reward[0]))
        memory.write_buffer(discreate_state(entry[0]), discreate_state(entry[1]), entry[2], reward)
        entry[0] = discreate_state(entry[0])
        entry[1] = discreate_state(entry[1])

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
            if(loss_itr == 1000):
                print("Loss:", loss.item(), loss_itr)
                agent.save_model("model")
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
