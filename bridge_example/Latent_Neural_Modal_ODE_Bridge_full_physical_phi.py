from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from datetime import datetime
from pathlib import Path
import os
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from models import RecognitionRNN, Recognition_q_logvar
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from interp1d import Interp1d
import scipy.io as sio
import networkx as nx
import matplotlib.font_manager as font_manager
import matplotlib.animation as animation

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

MEDIUM_SIZE = 17
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=13)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.ReLU())
    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)   

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val      
        
def loading_modal_paras(modal_dir, n_modes_used = 5):
    
    freq = sio.loadmat(modal_dir)["freq"][:n_modes_used]  
    phi =  sio.loadmat(modal_dir)["phi_aug"][:,:n_modes_used]  
    node_corr =  sio.loadmat(modal_dir)["node"]
    edges = sio.loadmat(modal_dir)["element"]
    
    omega = freq* 2 *np.pi
    xi = 0.5/100 * np.ones_like(omega)  
    
    omega = torch.from_numpy(omega).float()
    xi = torch.from_numpy(xi).float()  
    phi = torch.from_numpy(phi).float()    
    
    p = [omega, xi, phi]
    
    return  p, node_corr, edges 

def sliding_data(data_in, N = 2000, n = 500, mode = "sliding"):
    """Yield successive n-sized chunks from lst."""
    nt, nf = data_in.shape
    if mode == "sliding":
        ns = 20 # interval between two consecutive chunks
        # N = min(N, nt-n) 
        # print("N = " + str(N))
        data_out = []
        i = 0
        while i*ns + n <= nt:
            data_out.append(data_in[i*ns:i*ns + n,:])
            i += 1
        data_out = np.array(data_out)    
            
    elif mode == "batched":       
        N = min(nt // n, N)
        data_out = np.zeros((N, n, nf))
        for i in range(N):
            data_out[i,:,:] =  data_in[None, i*n:i*n + n,:]
          
    return data_out               

def loading_obs_data(data_dir, Nt):
             
    Obs_trajs = sio.loadmat(data_dir)["acc_measured"]

    dt =  float(sio.loadmat(data_dir)["dt"])
    
    Obs_trajs = sliding_data(Obs_trajs, N = 1500, n = 500, mode = "sliding")
        
    Obs_trajs = torch.from_numpy(Obs_trajs).float()  
        
    ts = np.linspace(0., dt*Nt - dt , num = Nt)
    ts = torch.from_numpy(ts).float()
    
    return  Obs_trajs, ts

def get_dof_index(node):    
    # xy = 0: x-direction
    # xy = 1: y-direction 
    node_number = node["node_number"]
    xy = node["xy"]
    
    dof_index = 3*( node_number - 1) + xy  
    
    return dof_index

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

def plot_resp(z_1, z_2, z_fem = None,
              label1 = "hybrid model", 
              label2 = "measured data",
              label3 = "unmeasured data",
              label_fem = "FEM",
              col_num = 1):
    # font = font_manager.FontProperties(family="Helvetica")
    plt.ioff()
    n_fig = z_1.shape[-1]
    Y_labels = ["A$_{"+str(i+1)+"}$" for i in range (n_fig)]
     
    fig = plt.figure(figsize=(8, 16))   
    for i in range(n_fig):
        plt.subplot( (n_fig+1)//col_num, (n_fig+1) // ((n_fig+1) // col_num), i+1) 

        if i != 1:
            plt.plot(z_2[:,i], '-', color = "silver", label = label2, lw = 2.5)
        else:
            plt.plot(z_2[:,i], "--", color = 'silver',label = label3, lw = 2.5)   
            
        plt.plot(z_1[:,i], '-', color = "blue", lw = 1, label = label1) 
        if z_fem is not None:
            plt.plot(z_fem[:,i], '-', color = "indianred", lw = 1, label = label_fem)
        
        plt.ylabel(Y_labels[i])
        
        if i == 0 or i == 1:
            plt.legend(loc = 1, ncol=3)
    plt.xlabel("$k$")
    plt.tight_layout()
    return fig

def plot_latent(z_1, 
              label1 = "latent quantities", 
              col_num = 5,
              mode = "latent"):
    plt.ioff()

    n_fig = z_1.shape[-1]
    
    Y_labels1 = [r"$q_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels2 = [r"$\dot{q}_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels = Y_labels1 + Y_labels2
   
    fig = plt.figure(figsize=(18, 8))   
    for i in range(n_fig):
        plt.subplot( (n_fig+1)//col_num, (n_fig+1) // ((n_fig+1) // col_num), i+1)           
        plt.plot(z_1[:,i], '-k',label = label1)     
        plt.ylabel(Y_labels[i])
        if i == 0:
            plt.legend()
    # plt.xlabel("$k$") 
    plt.tight_layout()
    return fig

def plot_graph(edges, node_corr, node_dis, a = 300.0, node_color = "k", edge_color = "k",label = "",):

    G1 = nx.Graph()
    G1.add_edges_from(edges)
    
    G2 = nx.Graph()
    G2.add_edges_from(edges)
    node_positions = {node: (node_corr[node-1,1],
                            node_corr[node-1,2] ) 
                      for node in G1.nodes}
    
    node_positions_new = {node: (node_corr[node-1,1] + a*node_dis[node-1,0],
                            node_corr[node-1,2] + a*node_dis[node-1,1]) 
                      for node in G2.nodes}
       
    nx.draw(G1,
                     pos = node_positions,
                     node_size = 30,
                     with_labels = False,
                     node_color = "grey",
                     edge_color = "grey",
                     ) 
    nx.draw(G2,
                     pos = node_positions_new,
                     node_size = 30,
                     with_labels = False,
                     node_color = node_color,
                     edge_color = edge_color,
                     label = label
                     ) 
    plt.draw() 
    plt.xlim([-3.3,3.3])
    plt.ylim([-0.3,1.1])
    return None
 
class NeuralModalODEfunc(nn.Module):

    def __init__(self, p, hidden_ndof = 2, obs_dim = 5,  hidden_dim = 128):
        
        super(NeuralModalODEfunc, self).__init__()
                
        z_dim = 2 * hidden_ndof
        self.n = hidden_ndof
            
        if p != ():
            print("modal-informed")
            Omega, Xi, Phi = p
            
            Phi = Phi / 4.6017e4
            
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
                                        nn.Linear(z_dim, hidden_dim, bias = False),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim, bias = False),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_ndof, bias = False),                                                                                  
                                        )
        
    def forward(self, t, zq):
                   
        qd = zq[...,self.n:]                       
        out1 = qd       
        out2 = self.trans_A(zq) + self.trans_net(zq)       
        # out2 = self.trans_A(zq) 

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
                 dof_indices, dof_indices_full,
                 obs_noise_std = 0.002
                 ):
    
    # loss_mse = nn.MSELoss()   
    pred_zq_sol = odeint(odefunc_model,
                         zq0,ts, 
                          method = "rk4" 
                         ).permute(1, 0, 2)
      
    pred_acc_sol = odefunc_model.trans_A(pred_zq_sol) + odefunc_model.trans_net(pred_zq_sol)  
          
    pred_dis =  odefunc_model.Phi_decoder(pred_zq_sol[...,:odefunc_model.n]) 
        
    pred_x_total = odefunc_model.Phi_decoder(pred_acc_sol) 
    pred_x = pred_x_total[..., dof_indices]
    pred_x_full = pred_x_total[..., dof_indices_full]
    
    noise_std_ = torch.zeros(pred_x.size()) + obs_noise_std
    noise_logvar = 2. * torch.log(noise_std_)
       
    logpx = log_normal_pdf(
         data_train, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(zq0.size())
    
    analytic_kl = normal_kl(zq0_mean, zq0_logvar,
                            pz0_mean, pz0_logvar).sum(-1)
    loss = torch.mean(-logpx + beta*analytic_kl, dim=0)
    
    return loss, pred_x_full, pred_x, pred_zq_sol, pred_dis  

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
    
    n_modes_used = 10
    n_dim = n_modes_used
    latent_dim = n_dim * 2
    rnn_len = 10
    obs_noise_std = 0.002
    batch_size = 16
    num_epochs = 1
    beta = 1.0
    a = 0.8
    lr = 1e-3
    Nt = 500 # number of time steps per batch
    encoder_type = "RNN_MLP" # "RNN_MLP"
    
    obs_types = ["acc", "dis+acc", "dis+vel"]
    obs_type = obs_types[0]
    print("obs_type = " + obs_type)
    print("encoder_tyep = " + encoder_type)
    
    hyper_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "beta": beta,
        "obs_type": obs_type,
        "rnn_len": rnn_len,
        "lr": lr,
        "encoder_type": encoder_type
    }  
            
    modal_dir =   "./data/modal_para.mat"
    data_dir1 =   "./data/acc_measured1.mat" 
    data_dir2 =   "./data/acc_measured2.mat" 
    data_dir3 =   "./data/acc_measured3.mat" 
    data_dir4 =   "./data/acc_measured4.mat" 
    data_dir5 =   "./data/acc_measured5.mat" 
    data_dir_fem = "./data/acc_FEM1.mat" 
    
    p, node_corr, edges = loading_modal_paras(modal_dir, n_modes_used = n_modes_used)
    
    Obs_trajs_full1, ts    = loading_obs_data(data_dir1,Nt)
    Obs_trajs_full2, _     = loading_obs_data(data_dir2,Nt)
    Obs_trajs_full3, _     = loading_obs_data(data_dir3, Nt)
    Obs_trajs_full4, _     = loading_obs_data(data_dir4, Nt)
    Obs_trajs_full5, _     = loading_obs_data(data_dir5, Nt)
    Obs_trajs_full_fem, _  = loading_obs_data(data_dir_fem, Nt)
    
    measured_nodes = [
                    {"name":"A1", "node_number":5,  "xy": 1},
                    {"name":"A2", "node_number":9,  "xy": 1},
                    {"name":"A3", "node_number":12, "xy": 1},
                    {"name":"A4", "node_number":15, "xy": 1},
                    {"name":"A5", "node_number":18, "xy": 1},
                    {"name":"A6", "node_number":21, "xy": 1},
                    {"name":"A7", "node_number":40, "xy": 0},
                    {"name":"A8", "node_number":53, "xy": 0},
                    ]
    nodes_full = [0,1,2,3,4,5,6,7]  
    nodes_for_train = [0,1,3,4,5,6,7] # 2 is left out for valiation

    Obs_trajs1 = Obs_trajs_full1[:,:,nodes_for_train]
    Obs_trajs2 = Obs_trajs_full2[:,:,nodes_for_train]
    Obs_trajs3 = Obs_trajs_full3[:,:,nodes_for_train]
    Obs_trajs4 = Obs_trajs_full4[:,:,nodes_for_train]
    Obs_trajs5 = Obs_trajs_full5[:,:,nodes_for_train]
    
    # get dof indices
    dof_indices = []
    for i in range(len(nodes_for_train)):
        dof_indices.append( get_dof_index(measured_nodes[nodes_for_train[i]]) )
           
    dof_indices_full = []
    for i in range(len(nodes_full)):
        dof_indices_full.append( get_dof_index(measured_nodes[nodes_full[i]]) )    
                  
    obs_dim =  Obs_trajs1.shape[-1]
    Obs_trajs_train = torch.cat((Obs_trajs1, Obs_trajs2,
                                 Obs_trajs3, Obs_trajs4, Obs_trajs5), dim = 0)
    Obs_trajs_test = Obs_trajs1
    
    Obs_trajs_full = torch.cat((Obs_trajs_full1, Obs_trajs_full2,
                                Obs_trajs_full3, Obs_trajs_full4, Obs_trajs_full5), dim = 0)
    Obs_trajs_full_test = Obs_trajs_full1
    
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
    loss_meter = RunningAverageMeter()    
    optimizer = optim.Adam(params, lr= lr)
    
    # create experiment directory and save config
    train_model_dir = "trained_models"
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(train_model_dir, dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)  
    
    load_model = "17-05-2022_15-47-59"

    if  load_model != "":       
        load_path =  os.path.join(train_model_dir, load_model,'checkpoint.pth')
        load_checkpoint(load_path)    
    
    global_step = 0
    epoch_loss = 0
    
    train_set = torch.utils.data.TensorDataset(Obs_trajs_train, 
                                               Obs_trajs_full,
                                               )    
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size= batch_size,
                                            shuffle=True)      
        
    for epoch in range(num_epochs):
        for _, (data_train, data_full) in enumerate(tqdm(train_loader)):
        
            global_step += 1            
            optimizer.zero_grad()
            
            # estimate zq0
            zq0, zq0_mean, zq0_logvar = sampling_zq0(odefunc,rec,data_train)
            
            loss, pred_x_full, pred_x, pred_zq_sol, pred_dis   = compute_loss(
                                                    odefunc, zq0, ts, data_train, 
                                                    dof_indices, dof_indices_full,
                                                     obs_noise_std = obs_noise_std)
            
            loss.backward()
            optimizer.step()       
            loss_meter.update(loss.item())
            
            # recording loss
            # epoch_loss += float(loss)
            batch_loss = loss / batch_size
        
        print(f"Epoch {epoch}: Training loss = {batch_loss.item():.6f}")
        print(f"Epoch {epoch}: Average ELBO = {-loss_meter.avg:.6f}")

        if  epoch % 50 == 0 :
            
            save_checkpoint(odefunc, rec, optimizer, epoch, loss, save_path)
            n_re = 0
            
            #-----------plotting train -----------------#
            
            fig_G = plt.figure(figsize=(7,7))
            for i in range(5):
                plt.subplot(5,1,i+1)
                pred_dis_reshape = pred_dis.reshape((pred_dis.shape[0],Nt,-1,3))[n_re,i,:,:]                               
                plot_graph(edges, node_corr, pred_dis_reshape.detach().numpy())
                plt.title("$k = " + str(i) +"$")
            plt.tight_layout()    
            fig_G.savefig(os.path.join(save_path, f"train_dis_full_{epoch:02d}.png"))
            plt.close(fig_G)
           
            fig0 = plot_resp(
                      pred_x_full.detach().numpy()[n_re,:,:], 
                      data_full.detach().numpy()[n_re,:,:],          
                      )
            fig0.savefig(os.path.join(save_path, f"train_full_{epoch:02d}.png"))
            plt.close(fig0)     
            
            fig1 = plot_latent(
                      pred_zq_sol.detach().numpy()[n_re,:,:], 
                      )
            fig1.savefig(os.path.join(save_path, f"train_latent_{epoch:02d}.png"))
            plt.close(fig1)
            
            #-----------plotting test -----------------#
            n_sample = 16
            
            Obs_trajs_sample = Obs_trajs_test[:n_sample,:,:]
            
            zq0, zq0_mean, zq0_logvar = sampling_zq0(odefunc,rec,Obs_trajs_sample)
            
            loss_test, pred_x_full_test, pred_x_test, pred_zq_sol_test, pred_dis_test = compute_loss(
                                odefunc, zq0, ts, Obs_trajs_sample, 
                                dof_indices, dof_indices_full,
                                obs_noise_std = obs_noise_std) 
            
            print('Epoch {}: Test loss = {:.6f}'.format(epoch, loss_test / n_sample))

            fig2= plot_resp(
                       pred_x_full_test.detach().numpy()[n_re,:,:], 
                      Obs_trajs_full_test.detach().numpy()[n_re,:,:], 
                      z_fem = Obs_trajs_full_fem.detach().numpy()[n_re,:,:], 
                      )
            fig2.savefig(os.path.join(save_path, f"test_full_{epoch:02d}.png"))
            plt.close(fig2)
            
            fig3 = plot_latent(
                      pred_zq_sol_test.detach().numpy()[n_re,:,:], 
                      )
            fig3.savefig(os.path.join(save_path, f"test_latent_{epoch:02d}.png"))
            plt.close(fig3)
            
            fig_G_test = plt.figure(figsize=(12.5,12.5))
            for i in range(5):
                plt.subplot(5,1,i+1)
                pred_dis_test_reshape = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,i,:,:]                               
                plot_graph(edges, node_corr, pred_dis_test_reshape.detach().numpy())
                plt.title("$k = " + str(i) +"$")
            plt.tight_layout()    
            fig_G_test.savefig(os.path.join(save_path, f"test_dis_full_{epoch:02d}.png"))
            plt.close(fig_G_test)

            plt.close("all")
            
            
            temp = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,:,:,:]                               
            temp2 = temp.detach().numpy()[:,39,0]
            plt.figure(figsize=(8,2))
            plt.plot(300*temp2, label = "Node 40")
            plt.ylabel("reconstructed displacement")
            plt.xlabel("$k$")
            plt.legend(loc = 1)
            plt.show()
                    
            u_fem = sio.loadmat("./data/deformation_FEM.mat")["u"]  
            u0_fem = u_fem[:,0].reshape(55,3)
            
            u0_hybrid = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,0,:,:].detach().numpy()  
            
            ratio = u0_hybrid[19,1] / u0_fem[19,1]
            
            fig = plt.figure(figsize=(15,3))                             
            plot_graph(edges, node_corr, u0_hybrid, 
                       node_color = "k", edge_color = "k",
                       label = "hybrid model")
            plot_graph(edges, node_corr, u0_fem*ratio, 
                       node_color = "indianred", edge_color = "indianred",
                       label = "FEM")
            plt.legend()
            plt.show()
            
            
            
                
                
                
                        
                        