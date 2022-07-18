# Neural Modal ODEs

This repository contains a PyTorch impelementation of a demonstrative example in the following paper:

* "Integrating Physics-based Modeling with Neural ODEs for Modeling High Dimensional Monitored Structures."
Zhilu Lai, Wei Liu, Xudong Jian, Kiran Bacsa, Limin Sun, and Eleni Chatzi (2022). 


## Framework

The architecture is comprised of:

* an encoder $\Psi_{\text{NN}}$ (performing inference from observational data of a handful of data to the initial conditions of latent variables $\textbf{z}_0$)
* Physics-informed Neural ODEs (Pi-Neural ODEs): modeling the dynamics of latent quantities
* a decoder $\Phi_p$: structred by the eigenmodes derived from eigen-analysis of linearized part 

![Graphical abstract of the framework](framework.png)


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
