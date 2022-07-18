# Neural Modal ODEs

This repository contains a PyTorch impelementation of a demonstrative example in the following paper:

* "Integrating Physics-based Modeling with Neural ODEs for Modeling High Dimensional Monitored Structures."
Zhilu Lai, Wei Liu, Xudong Jian, Kiran Bacsa, Limin Sun, and Eleni Chatzi (2022). 


## Framework

The architecture is comprised of:

* an encoder $\Psi_{\text{NN}}$ (performing inference from observational data of a handful of data to the initial conditions of latent variables $\textbf{z}_0$)
* Physics-informed Neural ODEs (Pi-Neural ODEs): modeling the dynamics of latent quantities
* a decoder $\Phi_p$: structured by the eigenmodes derived from eigen-analysis of linearized part of structural matrices of the physics-based models

![Graphical abstract of the framework](framework.png)


## Results
![prediction](fig/kn_0.5.png)
     

## Repository Overview
 * `data` - Generated data from the simulated system
   * `modal_para.npz` - Modal parameters for the simulated system.
   * `measured_data_nonlinear.npz` - Generated measurements.
 * `Neural_Modal_ODE_demo.py` - Manages training and evaluation of models.
 * `data_generation.py` - Generates data from a simulated 4-DOF nonlinear system.
 * `models.py` - PyTorch modules for the encoder.
