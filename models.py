import torch
import torch.nn as nn
# from torch.autograd.functional import jacobian, hessian


class Recognition_q(nn.Module):

    def __init__(self, latent_dim = 6, obs_dim = 3, nhidden=32):
        super(Recognition_q, self).__init__()
        
        self.o2h = nn.Linear(obs_dim//2, nhidden)
        self.h2q = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        
        h = torch.tanh(self.o2h(x))
        out = self.h2q(h)
        return out  
    
class Recognition_q_logvar(nn.Module):

    def __init__(self, latent_dim = 6, obs_dim = 3, nhidden=32):
        super(Recognition_q_logvar, self).__init__()
        
        n_dim = latent_dim // 2
        
        self.o2h = nn.Linear(obs_dim, nhidden)
        self.h2q = nn.Linear(nhidden, n_dim)

    def forward(self, x):
        
        h = torch.tanh(self.o2h(x))
        out = self.h2q(h)
        return out     
    

class RecognitionRNN(nn.Module):

    def __init__(self, out_dim = 4, obs_dim = 3, nhidden = 32):
        super(RecognitionRNN, self).__init__()
        
        self.h0 = nn.Parameter(torch.zeros(1, nhidden))
        
        self.nhidden = nhidden
 
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, out_dim)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, nbatch):
         
        out = self.h0.expand(nbatch,self.nhidden)
        return out

