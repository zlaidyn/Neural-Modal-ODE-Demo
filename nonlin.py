# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:58:37 2022

@author: Liu Wei
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os
from torchdiffeq import odeint
from models import RecognitionRNN, MLP
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#plt.ioff()
plt.rc('font', family='Times New Roman')
plt.rcParams["mathtext.fontset"] = "cm"

k00=np.load('./data/measured_data_kn_0.0.npz')["State_trajs"]
k05=np.load('./data/measured_data_kn_0.5.npz')["State_trajs"]
k10=np.load('./data/measured_data_kn_1.0.npz')["State_trajs"]

fig = plt.figure(figsize = (80,75) )   
plt.plot(k00[0,:,0],-(1*k00[0,:,8]+2*k00[0,:,9]+3*k00[0,:,10]+4*k00[0,:,11]), '-', color = "silver", label = '$k_n=0.0$', lw = 30)
plt.plot(k05[0,:,0],-(1*k05[0,:,8]+2*k05[0,:,9]+3*k05[0,:,10]+4*k05[0,:,11]), '-', color = "green", label = '$k_n=0.5$', lw = 30)
plt.plot(k10[0,:,0],-(1*k10[0,:,8]+2*k10[0,:,9]+3*k10[0,:,10]+4*k10[0,:,11]), '-', color = "blue", label = '$k_n=1.0$', lw = 30)
plt.xlabel("$x_1$", fontsize=200)
plt.ylabel('restoring force of the first DOF', fontsize=200)
plt.xticks(fontsize=200)
plt.yticks(fontsize=200)
#labels = ['unmeasured data', 'measured data', 'hybrid model']
fig.legend(loc='upper left', ncol=3, fontsize=180)
plt.tight_layout(rect=[0,0,1,0.9])
fig.savefig("./nonlin")
