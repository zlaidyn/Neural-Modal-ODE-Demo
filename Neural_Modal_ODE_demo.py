from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os
from torchdiffeq import odeint
from models import RecognitionRNN
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import networkx as nx

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
        
def loading_modal_paras(modal_dir, n_modes_used = 4):
    
    npzfile = np.load(modal_dir)
        
    omega = npzfile["omega"][:n_modes_used]  
    phi =  npzfile["phi_aug"][:,:n_modes_used]  
    xi = npzfile["xi"][:n_modes_used]  
    node_corr =  npzfile["node"]
    edges = npzfile["element"]   
     
    omega = torch.from_numpy(omega).float()
    xi = torch.from_numpy(xi).float()  
    phi = torch.from_numpy(phi).float()    
    
    p = [omega, xi, phi]
    
    return  p, node_corr, edges            

def loading_obs_data(data_dir):
    
    npzfile = np.load(data_dir)
             
    Obs_trajs = npzfile["Obs_trajs"]
    Obs_trajs_fem = npzfile["Obs_trajs_fem"]
    
    Nt = Obs_trajs.shape[1]

    dt = npzfile["dt"]
            
    Obs_trajs = torch.from_numpy(Obs_trajs).float()  
        
    ts = np.linspace(0., dt*Nt - dt , num = Nt)
    ts = torch.from_numpy(ts).float()
    
    return  Obs_trajs, Obs_trajs_fem, ts

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
              col_num = 2):
    # font = font_manager.FontProperties(family="Helvetica")
    plt.ioff()
    n_fig = z_1.shape[-1]
    Y_labels = ["A$_{"+str(i+1)+"}$" for i in range (n_fig)]
     
    fig = plt.figure(figsize = (12,6) )   
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
              col_num = 4,
              mode = "latent"):
    plt.ioff()

    n_fig = z_1.shape[-1]
    
    Y_labels1 = ["$q_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels2 = ["$\dot{q}_{"+str(i+1)+"}$" for i in range (n_fig//2)]
    Y_labels = Y_labels1 + Y_labels2
   
    fig = plt.figure(figsize = (12,6) )   
    for i in range(n_fig):
        plt.subplot( (n_fig+1)//col_num, (n_fig+1) // ((n_fig+1) // col_num), i+1)           
        plt.plot(z_1[:,i], '-k',label = label1)     
        plt.ylabel(Y_labels[i])
        if i == 0:
            plt.legend()
    # plt.xlabel("$k$") 
    plt.tight_layout()
    return fig

def plot_graph(edges, node_corr, node_dis, a = 1.0, node_color = "k", edge_color = "k",label = "",):

    G1 = nx.Graph()
    G1.add_edges_from(edges)
    
    G2 = nx.Graph()
    G2.add_edges_from(edges)
    node_positions = {node: (node_corr[node-1,1],
                            node_corr[node-1,2] ) 
                      for node in G1.nodes}
    
    node_positions_new = {node: (node_corr[node-1,1] + a*node_dis[node-1],
                            node_corr[node-1,2]) 
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
    return None
 
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
    loss = torch.mean(-logpx + analytic_kl, dim=0)

    experiment.log_metric("loss", loss, step=global_step)
    
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
    
    experiment = Experiment(
                api_key="xc4Txf2ZWPNveg4veZsy5Hxzt",
                project_name="Neural_Modal_ODE_demo",
                workspace="gamehere-007",
                )
    
    n_modes_used = 4
    n_dim = n_modes_used
    latent_dim = n_dim * 2
    rnn_len = 10
    obs_noise_std = 0.03
    batch_size = 16
    num_epochs = 10000
    lr = 1e-3
    encoder_type = "RNN_MLP" # "RNN_MLP"
    
    print("encoder_tyep = " + encoder_type)
    
    hyper_params = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "rnn_len": rnn_len,
        "lr": lr,
        "encoder_type": encoder_type
    }  
     
    experiment.log_parameters(hyper_params)    
        
    modal_dir =   "./data/modal_para.npz"
    data_dir =   "./data/measured_data.npz" 
    
    p, node_corr, edges = loading_modal_paras(modal_dir, n_modes_used = n_modes_used)
    
    Obs_trajs_full, Obs_trajs_full_fem, ts = loading_obs_data(data_dir)
    
    # nodes_full = [0,1,2,3]  
    dof_indices_full = [1,2,3,4]
    
    nodes_for_train = [0,2,3] 
    dof_indices = [1,3,4] # 2 is left out for valiation
    

    Obs_trajs = Obs_trajs_full[:,:,nodes_for_train]  
    N, Nt = Obs_trajs.shape[0], Obs_trajs.shape[1]
                  
    obs_dim =  Obs_trajs.shape[-1]
    Obs_trajs_train = Obs_trajs[:int(0.8*N),:,:]
    Obs_trajs_test = Obs_trajs[int(0.8*N):,:,:]
    
    Obs_trajs_full_train = Obs_trajs_full[:int(0.8*N),:,:]
    Obs_trajs_full_test = Obs_trajs_full[int(0.8*N):,:,:]
    Obs_trajs_full_fem_test  = Obs_trajs_full_fem[int(0.8*N):,:,:]
        
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
    
    load_model = ""

    if  load_model != "":       
        load_path =  os.path.join(train_model_dir, load_model,'checkpoint.pth')
        load_checkpoint(load_path)    
    
    global_step = 0
    epoch_loss = 0
    
    train_set = torch.utils.data.TensorDataset(Obs_trajs_train, 
                                               Obs_trajs_full_train,
                                               )    
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size= batch_size,
                                            shuffle=True)      
        
    with experiment.train():  
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
                
                # recording loss
                batch_loss = loss / batch_size
                experiment.log_metric("training_loss", batch_loss, step=global_step)                    

            if  epoch % 20 == 0 :
                
                save_checkpoint(odefunc, rec, optimizer, epoch, loss, save_path)
                print('Iter: {}, loss_train: {:.4f}'.format(epoch, batch_loss)) 
                n_re = 0
                #-----------plotting train -----------------#              
                fig_G = plt.figure()
                for i in range(5):
                    plt.subplot(1,5,i+1)
                    pred_dis_reshape = pred_dis[n_re,i,:]                               
                    plot_graph(edges, node_corr, pred_dis_reshape.detach().numpy())
                    plt.title("$k = " + str(i) +"$")
                plt.tight_layout()    
                experiment.log_figure(figure=fig_G, figure_name="train_dis_full{:02d}".format(epoch)) 
               
                fig0 = plot_resp(
                          pred_x_full.detach().numpy()[n_re,:,:], 
                          data_full.detach().numpy()[n_re,:,:],          
                          )
                experiment.log_figure(figure=fig0, figure_name="train_full_{:02d}".format(epoch)) 
                
                fig1 = plot_latent(
                          pred_zq_sol.detach().numpy()[n_re,:,:], 
                          )
                experiment.log_figure(figure=fig1, figure_name="train_latent_{:02d}".format(epoch)) 
                
                #-----------plotting test -----------------#
                n_sample = 16
                
                Obs_trajs_sample = Obs_trajs_test[:n_sample,:,:]
                
                zq0, zq0_mean, zq0_logvar = sampling_zq0(odefunc,rec,Obs_trajs_sample)
                
                loss_test, pred_x_full_test, pred_x_test, pred_zq_sol_test, pred_dis_test = compute_loss(
                                    odefunc, zq0, ts, Obs_trajs_sample, 
                                    dof_indices, dof_indices_full,
                                    obs_noise_std = obs_noise_std) 
                
                print('Iter: {}, loss_test: {:.4f}'.format(epoch, loss_test/n_sample)) 
                experiment.log_metric("test_loss", loss_test/n_sample, step = global_step)   

                fig2= plot_resp(
                           pred_x_full_test.detach().numpy()[n_re,:,:], 
                          Obs_trajs_full_test.detach().numpy()[n_re,:,:], 
                          z_fem = Obs_trajs_full_fem_test[n_re,:,:], 
                          )
                experiment.log_figure(figure=fig2, figure_name="test_full_{:02d}".format(epoch)) 
                                
                fig3 = plot_latent(
                          pred_zq_sol_test.detach().numpy()[n_re,:,:]
                          )
                experiment.log_figure(figure=fig3, figure_name="test_latent_{:02d}".format(epoch)) 
                             
                fig_G_test = plt.figure()
                for i in range(5):
                    plt.subplot(1,5,i+1)
                    pred_dis_test_reshape = pred_dis_test[n_re,i,:]                               
                    plot_graph(edges, node_corr, pred_dis_test_reshape.detach().numpy())
                    plt.title("$k = " + str(i) +"$")
                plt.tight_layout()    
                experiment.log_figure(figure=fig_G_test, figure_name="test_dis_full{:02d}".format(epoch))                                  
                plt.close("all")     
                             
                # if  epoch % 100 == 0: # generating videos
                #     print("saving figures ...")
                #     for i in range(Nt):
                #         pred_dis_test_reshape = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,i,:,:]                               
                #         fig = plt.figure(figsize=(15,3))
                #         plot_graph(edges, node_corr, pred_dis_test_reshape.detach().numpy())
                #         plt.savefig("./saved_fig/"+str(i)+".jpg")             
                #         plt.close("all") 
                        
                # temp = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,:,:,:]                               
                # temp2 = temp.detach().numpy()[:,39,0]
                # plt.figure(figsize=(8,2))
                # plt.plot(300*temp2, label = "Node 40")
                # plt.ylabel("reconstructed displacement")
                # plt.xlabel("$k$")
                # plt.legend(loc = 1)
                # plt.show()
                        
                # u_fem = sio.loadmat("./data/deformation_FEM.mat")["u"]  
                # u0_fem = u_fem[:,0].reshape(55,3)
                
                # u0_hybrid = pred_dis_test.reshape((pred_dis_test.shape[0],Nt,-1,3))[n_re,0,:,:].detach().numpy()  
                
                # ratio = u0_hybrid[19,1] / u0_fem[19,1]
                
                # fig = plt.figure(figsize=(15,3))                             
                # plot_graph(edges, node_corr, u0_hybrid, 
                #            node_color = "k", edge_color = "k",
                #            label = "hybrid model")
                # plot_graph(edges, node_corr, u0_fem*ratio, 
                #            node_color = "indianred", edge_color = "indianred",
                #            label = "FEM")
                # plt.legend()
                # plt.show()
                
                
                
                
                
                
                        
                        