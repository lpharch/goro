import torch
from network import QNetwork
from random import random
from random import randint
import os


def take_action(state, model):
    state_space  = 65
    action_space = 19
    action_scale = 6
    acc = []
    q_model = QNetwork(state_space, action_space, action_scale, 1, 1, 1, 0.9)
    checkpoint = torch.load((model), map_location=torch.device('cpu'))
    q_model.load_state_dict(checkpoint['modelA_state_dict'])
        
    out =  q_model(torch.tensor(state, dtype=torch.float))
    for tor in out:
        acc.append(torch.argmax(tor, dim=1)[[0]].item() )
    print("Actions suggested by the model ", model, acc)


models = os.listdir("./models/")
models = ["gem5modell4_density_600"]
for mod in models:
    print("------MODEL:", mod)
    for r in range(1000):
        state = [] 
        for s in range(65):
            state.append(random())
        take_action(state, "./models/"+mod)
