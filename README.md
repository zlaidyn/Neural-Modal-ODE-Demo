# Neural-Modal-ODE-Demo

This repository contains demo codes and data for the following working paper:
* Zhilu Lai, Wei Liu, Xudong Jian, Kiran Bacsa, Limin Sun, and Eleni Chatzi (2022). Integrating Physics-based Modeling with Neural ODEs for Modeling High Dimensional Monitored Structures.

In the scope of physics-informed machine learning, this paper proposes a framework for integrating physics-based modeling with deep learning (particularly, Neural Ordinary Differential Equations -- Neural ODEs) for modeling the dynamics of monitored and high-dimensional engineered systems.

## Framework
![framework](framework.png)

## Results
![prediction](https://s3.amazonaws.com/comet.ml/image_2d0c77c6b6bd4074bb9cc9cc1d2e4d2f-zIJaeUdWdM8SHeZ4f1O4Q1mBd.svg)

The corresponding predictions of displacements, velocities and accelerations are shown in the above figure, denoted by the blue lines. This prediction is compared with the actual measurements in grey color and predictions by the FEM models in red color. One can see that the FEM model offers satisfactory results, while some channel predictions are out of phase and fail to accurately follow the actual measurement, most possibly due to the inaccurate modeling of damping. The prediction from the proposed hybrid model is evidently more accurate than the FEM model, almost aligning with the actual measurements.

It is noted that only the displacements of the fourth channel and the accelerations of the first, third and fourth channels are used for training the hybrid model, denoted by solid grey lines, while all other data is unmeasured and not involved in training process. The predictions for the unmeasured data come from the full-order reconstructed responses. One can see that the reconstruction of them still highly agrees with the actual data, even though they are not used for the training.      

## Repository Overview
 * `data` - Generated data from the simulated system
   * `modal_para.npz` - Modal parameters for the simulated system.
   * `measured_data_nonlinear.npz` - Generated measurements.
 * `Neural_Modal_ODE_demo.py` - Manages training and evaluation of models.
 * `data_generation.py` - Generates data from a simulated 4-DOF nonlinear system.
 * `models.py` - PyTorch modules for the encoder.
