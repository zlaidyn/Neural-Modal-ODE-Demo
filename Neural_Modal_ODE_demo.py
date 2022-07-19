# =============================================================================
# This is a Pytorch implementation of Neural Modal ODEs
# A demonstraive example of a 4-DOF linear/nonlinear Structural Dynamical Systems
#
# Code Authors: Zhilu Lai, Liu Wei, Kiran Bacsa @ ETH and Singapore-ETH Centre
# Reference: "Integrating Physics-based Modeling with Neural ODEs for 
#             Modeling High Dimensional Monitored Structures." 
#             Zhilu Lai, Wei Liu, Xudong Jian, Kiran Bacsa, 
#             Limin Sun, and Eleni Chatzi (2022)
# =============================================================================

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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def loading_modal_paras(modal_dir, n_modes_used = 4):
    
    npzfile = np.load(modal_dir)
        
    omega = npzfile["omega"][:n_modes_used]  
    phi =  npzfile["phi"][:,:n_modes_used]  
    xi = npzfile["xi"][:n_modes_used]  
    node_corr =  npzfile["node"]
    edges = npzfile["element"]   
     
    omega = torch.from_numpy(omega).float()
    xi = torch.from_numpy(xi).float()  
    phi = torch.from_numpy(phi).float()    
    
    p = [omega, xi, phi]
    
    return  p, node_corr, edges            

def loading_obs_data(data_dir, obs_idx):
    
    npzfile = np.load(data_dir)            
    State_trajs = npzfile["State_trajs"]
    State_trajs_fem = npzfile["State_trajs_fem"]
    dt = npzfile["dt"]
    
    State_trajs = torch.from_numpy(State_trajs).float()    
    Obs_trajs = State_trajs[:,:, obs_idx]
    
    Nt = Obs_trajs.shape[1]
                    
    ts = np.linspace(0., dt*Nt - dt , num = Nt)
    ts = torch.from_numpy(ts).float()
    
    return  Obs_trajs, State_trajs, State_trajs_fem, ts

def log_normal_pdf(x, mean, logvar):
    
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def plot_resp(z_1, z_2, obs_idx, z_fem = None,
              label1 = "hybrid model", 
              label2 = "measured data",
              label3 = "unmeasured data",
              label_fem = "FEM",
              col_num = 4):
    plt.ioff()
    plt.rc('font', family='Times New Roman')
    plt.rcParams["mathtext.fontset"] = "cm"
    n_fig = z_1.shape[-1]
    Y_label1 = [r"$x_{"+str(i+1)+"}$" for i in range (n_fig//3)]
    Y_label2 = [r"$\dot{x}_{"+str(i+1)+"}$" for i in range (n_fig//3)]
    Y_label3 = [r"$\ddot{x}_{"+str(i+1)+"}$" for i in range (n_fig//3)]
    
    Y_labels = Y_label1 + Y_label2 + Y_label3
    
    fig = plt.figure(figsize = (140,75) )   
    for i in range(n_fig):
        plt.subplot( (n_fig+1)//col_num, (n_fig+1) // ((n_fig+1) // col_num), i+1) 

        if i in obs_idx:
            plt.plot(z_2[:,i], '-', color = "silver", label = label2, lw = 30)
        else:
            plt.plot(z_2[:,i], "--", color = 'silver',label = label3, lw = 30)   
            
        plt.plot(z_1[:,i], '-', color = "blue", lw = 10, label = label1) 
        if z_fem is not None:
            plt.plot(z_fem[:,i], '-', color = "indianred", lw = 10, label = label_fem)
        
        plt.ylabel(Y_labels[i], fontsize=200)
        
        if i in range(8,12):
            plt.xlabel("$k$", fontsize=200)
        else:
            plt.xticks([])
        plt.xticks(fontsize=200)
        plt.yticks(fontsize=200)

    if z_fem is not None:
        lines = [Line2D([0], [0], color="silver", linewidth=20, linestyle='--'),
                 Line2D([0], [0], color="silver", linewidth=20, linestyle='-'),
                 Line2D([0], [0], color="indianred", linewidth=20, linestyle='-'),
                 Line2D([0], [0], color="blue", linewidth=20, linestyle='-')]
        labels = ['unmeasured data', 'measured data', 'FEM', 'hybrid model']
    else:
        lines = [Line2D([0], [0], color="silver", linewidth=20, linestyle='--'),
                 Line2D([0], [0], color="silver", linewidth=20, linestyle='-'),
                 Line2D([0], [0], color="blue", linewidth=20, linestyle='-')]
        labels = ['unmeasured data', 'measured data', 'hybrid model']
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0,1), ncol=4, fontsize=180)
    plt.tight_layout(rect=[0,0,1,0.9])
    return fig

def plot_latent(z_1, 
              label1 = "latent quantities", 
              col_num = 4,
              mode = "latent"):
    plt.ioff()

    n_fig = z_1.shape[-1]
    
    Y_labels1 = ["$q_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels2 = ["$\dot{q}_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels = Y_labels1 + Y_labels2
   
    fig = plt.figure(figsize = (14,6) )   
    for i in range(n_fig):
        plt.subplot( (n_fig+1)//col_num, (n_fig+1) // ((n_fig+1) // col_num), i+1)           
        plt.plot(z_1[:,i], '-k',label = label1)     
        plt.ylabel(Y_labels[i])
        if i == 0:
            plt.legend()
    plt.tight_layout()
    return fig
 
class NeuralModalODEfunc(nn.Module):

    def __init__(self, p, hidden_ndof = 2, obs_dim = 5,  hidden_dim = 128):
        
        super(NeuralModalODEfunc, self).__init__()
                
        z_dim = 2 * hidden_ndof
        self.n = hidden_ndof
            
        if p != ():
            print("modal-informed")
            Omega, Xi, Phi = p
            
            Phi_dim = Phi.shape[0]
            
            Omega = Omega.squeeze(-1)
            Xi = Xi.squeeze(-1)
            
            # A linear MLP            
            K = torch.diag_embed(Omega**2)
            C = torch.diag_embed(2 * Xi * Omega) 
            
            A = np.concatenate([-K, -C], axis=1)
    
            A = torch.from_numpy(A).float()
                  
            self.trans_A = nn.Linear(z_dim, hidden_ndof, bias = False)
            self.trans_A.weight.data = A
            for param in self.trans_A.parameters():
                param.requires_grad = False 
                             
            self.Phi_decoder = nn.Linear(hidden_ndof, Phi_dim, bias = False)
            self.Phi_decoder.weight.data = Phi
            for param in self.Phi_decoder.parameters():
                param.requires_grad = False 
                
        else:           
            self.trans_A = 0
            
        self.Phi_net_encoder = MLP(obs_dim,
                                    [hidden_dim, hidden_dim],
                                    z_dim)
        
        self.trans_net = nn.Sequential(
                                        nn.Linear(z_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_ndof),                                                                                  
                                        )
        
    def forward(self, t, zq):
                   
        qd = zq[...,self.n:]                       
        out1 = qd       
        out2 = self.trans_A(zq) + self.trans_net(zq)       

        zq_dot = torch.cat([out1,out2], axis = -1)
        
        return zq_dot    
    
def sampling_zq0_RNN_MLP(odefunc_model, rec_model,data_train):
    
    x0 = data_train[:,0,:]               
    nbatch = x0.shape[0]    
    q0= odefunc_model.Phi_net_encoder(x0)
    q0_mean, q0_logvar = q0[:,:n_dim], q0[:,n_dim:]
    
    h = rec_model.initHidden(nbatch = nbatch)
    for t in reversed(range(rnn_len)):
        obs = data_train[:, t, :]
        out, h = rec_model.forward(obs, h)
        
    qd0_mean, qd0_logvar = out[:, :n_dim], out[:, n_dim:]
    
    zq0_mean = torch.cat( (q0_mean, qd0_mean), axis = 1)
    zq0_logvar = torch.cat( (q0_logvar, qd0_logvar), axis = 1)
    distrib = MultivariateNormal(loc = zq0_mean, 
                          covariance_matrix= torch.diag_embed(torch.exp(zq0_logvar)))
    zq0 = distrib.rsample()
    
    return zq0, zq0_mean, zq0_logvar    
    
def sampling_zq0_RNN(odefunc_model, rec_model,data_train):
             
    nbatch = data_train.shape[0]    
    h = rec_model.initHidden(nbatch = nbatch)
    for t in reversed(range(rnn_len)):
        obs = data_train[:, t, :]
        out, h = rec_model.forward(obs, h)
        
    zq0_mean, zq0_logvar = out[:, :2*n_dim], out[:, 2*n_dim:] 
    distrib = MultivariateNormal(loc = zq0_mean, 
                          covariance_matrix= torch.diag_embed(torch.exp(zq0_logvar)))
    zq0 = distrib.rsample()
    
    return zq0, zq0_mean, zq0_logvar

def compute_loss(odefunc_model, zq0, ts, data_train, 
                 obs_idx, obs_noise_std = 0.002):
    
    pred_zq_sol = odeint(odefunc_model,
                         zq0,ts, 
                          method = "rk4" 
                         ).permute(1, 0, 2)
      
    pred_acc_sol = odefunc_model.trans_A(pred_zq_sol) + odefunc_model.trans_net(pred_zq_sol) 
            
    pred_dis =  odefunc_model.Phi_decoder(pred_zq_sol[...,:odefunc_model.n]) 
    pred_vel = odefunc_model.Phi_decoder(pred_zq_sol[...,odefunc_model.n:])        
    pred_acc = odefunc_model.Phi_decoder(pred_acc_sol) 
    
    pred_state = torch.cat([pred_dis, pred_vel, pred_acc], dim = -1) 
    pred_x = pred_state[...,obs_idx]
    
    noise_std_ = torch.zeros(pred_x.size()) + obs_noise_std
    noise_logvar = 2. * torch.log(noise_std_)
       
    logpx = log_normal_pdf(
         data_train, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(zq0.size())
    
    analytic_kl = normal_kl(zq0_mean, zq0_logvar,
                            pz0_mean, pz0_logvar).sum(-1)
    loss = torch.mean(-logpx + analytic_kl, dim=0)
    
    return loss, pred_x, pred_state, pred_zq_sol 

def save_checkpoint(odefunc, rec, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'func_state_dict': odefunc.state_dict(),
        'rec_state_dict': rec.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,      
    }, os.path.join(save_path, 'checkpoint.pth'))
    print('saving ckpt at {}'.format(save_path))    
    
def load_checkpoint(load_path):
    # assert exists(cfg.load_model), \
    #     "--load-model and/or --load-opt misspecified"
    checkpoint = torch.load(load_path)
    
    odefunc.load_state_dict(checkpoint['func_state_dict'])
    rec.load_state_dict(checkpoint['rec_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    print('Loaded ckpt from {}'.format(load_path))    

if __name__ == '__main__': 
    
    
    n_modes_used = 4
    n_dim = n_modes_used
    latent_dim = n_dim * 2
    rnn_len = 10
    obs_noise_std = 0.03
    batch_size = 16
    num_epochs = 200
    lr = 1e-3
    encoder_type = "RNN_MLP" #  two types： "RNN" and "RNN_MLP"
    
    print("encoder_tyep = " + encoder_type)
    
    obs_idx = [3,8,10,11] # dis - 0,1,2,3; vel - 4,5,6,7; acc - 8,9,10,11        
        
    modal_dir =   "./data/modal_para.npz"
    data_dir =   "./data/measured_data_kn_0.5.npz" # kn = 0, 0.5, or 1 for different levels of nonlinearity
      
    p, node_corr, edges = loading_modal_paras(modal_dir, n_modes_used = n_modes_used)
    
    Obs_trajs, State_trajs, State_trajs_fem, ts = loading_obs_data(data_dir, obs_idx)

    N, Nt, obs_dim = Obs_trajs.shape
    
    Obs_trajs_train = Obs_trajs[:int(0.8*N),:,:]
    Obs_trajs_test = Obs_trajs[int(0.8*N):,:,:]
    
    State_trajs_train = State_trajs[:int(0.8*N),:,:]
    State_trajs_test = State_trajs[int(0.8*N):,:,:]
    
    State_trajs_fem_train = State_trajs_fem[:int(0.8*N),:,:]
    State_trajs_fem_test = State_trajs_fem[int(0.8*N):,:,:]   
        
    odefunc = NeuralModalODEfunc(p, hidden_ndof = n_dim, obs_dim = obs_dim)
    
    if encoder_type == "RNN":
        rec = RecognitionRNN(out_dim = 2*n_dim + 2*n_dim, obs_dim = obs_dim)
        sampling_zq0 = sampling_zq0_RNN
    elif encoder_type == "RNN_MLP":
        rec = RecognitionRNN(out_dim = 2*n_dim, obs_dim = obs_dim)
        sampling_zq0 = sampling_zq0_RNN_MLP
    else:
        raise("error")
       
    params = (
                list(odefunc.parameters()) + 
                list(rec.parameters())               
             )
    optimizer = optim.Adam(params, lr = lr)
    
    # create experiment directory and save config
    train_model_dir = "trained_models"
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(train_model_dir, dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)  
    
    # load_model = "31-05-2022_05-51-49_kn_1"  # load pre-trained models
    load_model = ""

    if  load_model != "":       
        load_path =  os.path.join(train_model_dir, load_model,'checkpoint.pth')
        load_checkpoint(load_path)    
    
    global_step = 0
    epoch_loss = 0
    
    train_set = torch.utils.data.TensorDataset(Obs_trajs_train, 
                                               State_trajs_train,
                                               )    
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size= batch_size,
                                            shuffle=True)      
    

    for epoch in range(num_epochs):
        for _, (data_train, state_full) in enumerate(tqdm(train_loader)):
    
            global_step += 1            
            optimizer.zero_grad()
                
            # estimate zq0
            zq0, zq0_mean, zq0_logvar = sampling_zq0(odefunc,rec,data_train)

            loss, pred_x, pred_state, pred_zq_sol  = compute_loss(
                odefunc, zq0, ts, data_train, 
                obs_idx, obs_noise_std = obs_noise_std)

            loss.backward()
            optimizer.step()       

            # recording loss
            batch_loss = loss / batch_size

        if  epoch % 5 == 0 :

            save_checkpoint(odefunc, rec, optimizer, epoch, loss, save_path)
            print('Epoch: {}, loss_train: {:.4f}'.format(epoch, batch_loss)) 
            n_re = 0
            #-----------plotting train -----------------#                             
            fig0 = plot_resp(
                pred_state.detach().numpy()[n_re,:,:], 
                state_full.detach().numpy()[n_re,:,:], 
                obs_idx
                )
            fig0.savefig("fig/train_full_{:02d}".format(epoch))
            
            fig1 = plot_latent(
                pred_zq_sol.detach().numpy()[n_re,:,:], 
                )
            fig1.savefig("fig/train_latent_{:02d}".format(epoch))

            #-----------plotting test -----------------#
            n_sample = 16

            Obs_trajs_sample = Obs_trajs_test[:n_sample,:,:]

            zq0, zq0_mean, zq0_logvar = sampling_zq0(odefunc,rec,Obs_trajs_sample)

            loss_test, pred_x_test, pred_state_test, pred_zq_sol_test = compute_loss(
                odefunc, zq0, ts, Obs_trajs_sample, 
                obs_idx, obs_noise_std = obs_noise_std) 

            print('Epoch: {}, loss_test: {:.4f}'.format(epoch, loss_test/n_sample)) 

            fig2= plot_resp(
                pred_state_test.detach().numpy()[n_re,:,:], 
                State_trajs_test.detach().numpy()[n_re,:,:], 
                obs_idx,
                z_fem = State_trajs_fem_test[n_re,:,:], 
                )
            fig2.savefig("fig/test_full_{:02d}".format(epoch))

            fig3 = plot_latent(
                pred_zq_sol_test.detach().numpy()[n_re,:,:]
                )
            fig3.savefig("fig/test_latent_{:02d}".format(epoch))

            plt.close("all")     
                             