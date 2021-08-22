import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 



 
path="/home/ml/test/BipedalWalker-BranchingDQN/csvs/" 
fcsvs = os.listdir(path)
idx=0
print("making the file")
for csv in fcsvs:
    data = pd.read_csv(path+csv, header=None) 
    print("File ", csv)
    for column in range(7, 89):
        # data[column] = (data[column] - data[column].mean()) / data[column].std() 
        data[column] = (((data[column] - data[column].min()) / (data[column].max() - data[column].min()))*100)
        data[column] = data[column].fillna(0)    
    # for column in data:
        # data[column]=data[column].astype(float)
        # print("col", column)
        # print(data[column].dtypes)
        # if(data[column].dtypes=="float64"):
            # data[column]=data[column].astype(int)
    # print(data[89])
    print("Writing to file-------------------", idx)
    # print("Writing to file-------------------", data[89])
    # print("Writing to file-------------------", data.dtypes)
    idx+=1
    # data=data.astype('float64')
    # data=data.astype('int32')
    data.to_csv('intset.csv', index=False, mode='a', header=None)  
    # break