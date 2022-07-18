# Neural Modal ODEs

This repository contains a PyTorch impelementation of a demonstrative example in the following paper:

* "Integrating Physics-based Modeling with Neural ODEs for Modeling High Dimensional Monitored Structures."
Zhilu Lai, Wei Liu, Xudong Jian, Kiran Bacsa, Limin Sun, and Eleni Chatzi (2022). 
In the scope of physics-informed machine learning, this paper proposes a framework for integrating physics-based modeling with deep learning (particularly, Neural Ordinary Differential Equations -- Neural ODEs) for modeling the dynamics of monitored and high-dimensional engineered systems.

## Framework
![framework](framework.png)

We summarize the proposed architecture in the above flowchart, which concatenates an encoder $\Psi_{\text{NN}}$ and a decoder $\Phi_p$, with  Physics-informed Neural ODEs (Pi-Neural ODEs).
The role of the encoder is to perform inference from observational data of a handful of DOFs to the initial conditions of latent variables $\textbf{z}_0$.
The evolution of the dynamics initiating from $\textbf{z}_0$ is learned and modeled by means of Physics-informed Neural ODEs, where the physics-informed term adopts a modal representation derived from the physics-based model. 
The prediction of $\textbf{z}_0, \textbf{z}_1, ... ,\textbf{z}_t, ... ,  \textbf{z}_T$ at time step $t_0, t_1, ... ,t, ... ,  t_T$,  obtained from the previous step is mapped back to the original observations space via the decoder $\textbf{x}_t = \Phi_p(\textbf{z}_t)$  $(t = 0,1,...,T)$. This is then compared against the actually obtained measurements to minimize the prediction error, which effectuates the training of the proposed model. In what follows, we offer the details of the formulation of the three outlined components (encoder, Pi-Neural ODEs, and decoder) to the suggested framework.

## Results
![prediction](fig/kn_0.5.png)

The corresponding predictions of displacements, velocities and accelerations are shown in the above figure, denoted by the blue lines. This prediction is compared with the actual measurements in grey color and predictions by the FEM models in red color. One can see that the FEM model offers satisfactory results, while some channel predictions are out of phase and fail to accurately follow the actual measurement, most possibly due to the inaccurate modeling of damping. The prediction from the proposed hybrid model is evidently more accurate than the FEM model, almost aligning with the actual measurements.

It is noted that only the displacements of the fourth channel and the accelerations of the first, third and fourth channels are used for training the hybrid model, denoted by solid grey lines, while all other data is unmeasured and not involved in training process. The predictions for the unmeasured data come from the full-order reconstructed responses. One can see that the reconstruction of them still highly agrees with the actual data, even though they are not used for the training.      

## Repository Overview
 * `data` - Generated data from the simulated system
   * `modal_para.npz` - Modal parameters for the simulated system.
   * `measured_data_nonlinear.npz` - Generated measurements.
 * `Neural_Modal_ODE_demo.py` - Manages training and evaluation of models.
 * `data_generation.py` - Generates data from a simulated 4-DOF nonlinear system.
 * `models.py` - PyTorch modules for the encoder.
