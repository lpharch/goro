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



data = pd.read_csv('./intset.csv', header=None) 
data=data.iloc[:,7:48]
res=data.corr(method ='pearson')
res.to_csv("cor.csv", index=False)
