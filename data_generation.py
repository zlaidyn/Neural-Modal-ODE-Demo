import numpy as np
from scipy.integrate import odeint as odeint_0
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import signal

def build_MKC(p):
    
    m1, m2, m3, m4, k1, k2, k3, k4, c1,c2, c3, c4 = p
    
    M = np.diag([m1,m2,m3,m4])   
    C = np.diag([c1,c2,c3,c4])  
    K = np.array([[k1+k2, -k2, 0, 0.0],
                  [-k2, k2+k3,-k3,  0],
                  [0, -k3, k3+k4, -k4 ],
                  [0, 0, -k4, k4 ],
                  ])
    return M, K, C    

def model_4dof(z, t, p):
    
    M,K,C = build_MKC(p)
    
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-np.linalg.solve(M, K), -np.linalg.solve(M, C)], axis=1),  # movement equations
        ], axis=0)
    
    dz = A @ z
    return dz        
       
def generate_model_4dof(
                        z0, tspan,
                        m1 = 1.0, m2 = 2.0, m3 = 3.0, m4 = 4.0,
                        k1 = 1.0, k2 = 2.0, k3 = 3.0, k4 = 4.0,
                        c1 = 0.1, c2 = 0.1, c3 = 0.1, c4 = 0.1,
                        ): 
    
    # run simulation
    nt = len(tspan)      
    p = [m1, m2, m3, m4, k1, k2, k3, k4, c1, c2, c3, c4]
    
    z_sol = odeint_0(model_4dof, z0, tspan, args=(p,),
                      # atol=abserr, rtol=relerr
                      )
    
    z_sol_dot = np.zeros_like(z_sol)
    for i in range(nt):
        z_sol_dot[i,:] = model_4dof(z_sol[i,:],0,p)
                        
    return z_sol, z_sol_dot, p

def compute_eig(p, normalized = True):
    
    M, K, C = build_MKC(p)
    n_dof = 4
    
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-np.linalg.solve(M, K), -np.linalg.solve(M, C)], axis=1),  # movement equations
        ], axis=0)    
 
    lambda_i, phi_i = np.linalg.eig(A)
    
    omega_i = np.abs(lambda_i) 
    ix = np.argsort(np.abs(lambda_i))
    xi = -np.real(lambda_i)/np.abs(lambda_i) # damping ratio
     
    xi_sorted = xi[ix[::2]]
    phi_sorted = np.real(phi_i[n_dof:,ix[::2]])
    omega_sorted = omega_i[ix[::2]]
    
    phi_normalized = np.zeros_like(phi_sorted)
    if normalized == True:
       for i in range(n_dof): 
           phi_normalized[:,i] =  phi_sorted[:,i] / np.max(np.abs(phi_sorted[:,i]))
       phi_sorted = phi_normalized    
    return phi_sorted, omega_sorted, xi_sorted

def plot_mode_shapes(phi):
    
    phi_aug = np.zeros((phi.shape[0]+1, phi.shape[1]))
    
    for i in range(n_dof):
        plt.subplot(1,n_dof,i+1)
        temp = np.hstack( (0, phi[:,i] )) 
        plt.plot( temp , np.arange(n_dof+1), 
                 ".-",
                 markersize= 10)
        plt.title(r"$\phi_"+str(i)+"$")
        phi_aug[:,i] = temp
        
    return phi_aug    

def reconstruction(z0, Phi, omega, xi):
    
    n_phi = Phi.shape[1]
    n_dof = z0.shape[0] // 2

    temp =  np.matmul(
                        np.linalg.inv ( np.matmul( Phi.T , Phi) ),  
                        Phi.T 
                        )
            
    q0 = np.matmul(temp, z0[:n_dof])
    qd0 = np.matmul(temp, z0[n_dof:])
    
    zq0 = np.hstack( (q0,qd0) )    

    zq_sol = odeint_0(modal_ode, zq0, tspan, args=(omega, xi))
    zq_sol_dot = np.zeros_like(zq_sol)

    for i in range(nt):
        zq_sol_dot[i,:] = modal_ode(zq_sol[i,:],0,omega, xi)
       
    z_recon = np.zeros_like(zq_sol)
    
    z_recon_dot = np.zeros_like(zq_sol_dot)
    
    for i in range(len(zq_sol)):
        z_recon[i,:n_dof] = np.matmul(Phi,zq_sol[i,:n_phi])
        z_recon[i,n_dof:] = np.matmul(Phi,zq_sol[i,n_phi:])
        
        z_recon_dot[i,:n_dof] = np.matmul(Phi,zq_sol_dot[i,:n_phi])
        z_recon_dot[i,n_dof:] = np.matmul(Phi,zq_sol_dot[i,n_phi:])
        
    return z_recon, z_recon_dot   

def modal_ode(q,t,omega, xi):
    
    n_dof = omega.shape[0]
    C_ = np.diag(2*xi*omega)  
    K_ = np.diag(omega**2)
    
    A = np.concatenate(
        [
            np.concatenate([np.zeros([n_dof, n_dof]), np.eye(n_dof)], axis=1),  # link velocities
            np.concatenate([-K_, -C_], axis=1),  # movement equations
        ], axis=0)       

    dq = A @ q 
    return dq 

def plot_comp(z_1, z_2, label1 = "", label2 = ""):
    Y_labels = ["$z_1$", "$z_2$", "$z_3$","$z_4$",
            "$\dot{z}_1$", "$\dot{z}_2$", "$\dot{z}_3$","$\dot{z}_4$"
            ]
    for i in range(8):
        plt.subplot(2,4,i+1)
        
        plt.plot(z_2[:,i], '.r',label = label2, markersize= 3 )
        plt.plot(z_1[:,i], '-k',label = label1)    
        
        plt.ylabel(Y_labels[i])
    plt.legend()
    plt.tight_layout()
    return None

def add_noise(data, noise_factor):
    
    noise = np.std(data) * noise_factor
    obs_noise = data + noise * np.random.randn(*data.shape)     
    return obs_noise

if __name__ == '__main__':
 
    n_dof = 4    
    t_max = 50.0
    dt = 0.1
    fs = 1/dt
    nt = int(t_max/dt) +1
    tspan = np.linspace(0., t_max , num = nt )
    
    N = 300 # number of realizations
    noise_factor = 0.03
    State_trajs = np.zeros((N,nt,n_dof*3))
    Obs_trajs = np.zeros((N,nt,n_dof))
    # obs_idx = [8,9,10,11] # measuring the acc
    
    for i in range(N):
        print("n = {}/{}".format(i,N))
        
        z0 = np.random.randn(n_dof*2,)
        z_sol, z_sol_dot, p = generate_model_4dof(z0, tspan)                
      
        State_trajs[i,:,:] = np.concatenate([z_sol, z_sol_dot[:, n_dof:]], axis=1)
        obs = State_trajs[i,:,2*n_dof:]
        obs_noise = add_noise(obs,noise_factor)  
        Obs_trajs[i,:,:] = obs_noise
                
    phi_sorted, omega_sorted, xi_sorted = compute_eig(p)
    phi_sorted_noise =  add_noise(phi_sorted,0.03)
    omega_sorted_noise =  add_noise(omega_sorted,0.05)
    xi_sorted_noise =  add_noise(xi_sorted,noise_factor)
        
    plt.figure()
    _ = plot_mode_shapes(phi_sorted)
    phi_aug = plot_mode_shapes(phi_sorted_noise)
       
    z0_new = State_trajs[0,0,:2*n_dof]
    xi_fem = 1.0/100 * np.ones_like(omega_sorted_noise) 
    
    z_fem, z_fem_dot =  reconstruction(z0_new, phi_sorted_noise, omega_sorted_noise, xi_fem)
    plt.figure()    
    plot_comp(z_fem, State_trajs[0, :, :2*n_dof],               
                  label1 = "Prior FEM model",
                  label2 = "measured data"
                  )
    
    plt.figure()    
    plot_comp(z_fem_dot, State_trajs[0, :, n_dof:],               
                  label1 = "Prior FEM model",
                  label2 = "measured data"
                  )
            
    State_trajs_fem =  np.zeros_like(State_trajs)
    
    for i in range(N):
        z0_new = State_trajs[i,0,:2*n_dof]        
        z_fem, z_fem_dot =  reconstruction(z0_new, phi_sorted_noise, omega_sorted_noise, xi_fem) 
        State_trajs_fem[i,:,:] = np.concatenate([z_fem, 
                                             z_fem_dot[:, n_dof:]], axis=1)

        
    
    element = np.array([[1,2],
                        [2,3],
                        [3,4],
                        [4,5]
                        ])
    node = np.array([[1,0,0],
                    [2,0,1],
                    [3,0,2],                   
                    [4,0,3],
                    [5,0,4.0],                 
                    ])
       
    np.savez("./data/measured_data.npz",            
                Obs_trajs = Obs_trajs, 
                dt = dt,
                State_trajs = State_trajs,  
                State_trajs_fem = State_trajs_fem
                )

    np.savez("./data/modal_para.npz",            
                phi_aug = phi_aug, 
                omega = omega_sorted_noise,
                xi = xi_fem,
                element = element,
                node = node
                )    