# RenOnet: Renormalized Operator networks for complex social systems

This repository contains an implementation of a multiscale operator learning framework for complex social systems. This framework sequentially learns a multiscale structure for a complex system given an initial adjacency matrix $A^{(0)}$ and length $K$ history of the system $`\{u_{i,t-k}^{(0)}, \ k=1,\ldots,K \}`$  at various evaluation times $t$. The model then predicts both the value of the system at each scale $\tilde{u}^{(l)}$ as well as infers the operator of the governing dynamics (see figure below).

A brief overview of important modules in this repository are:

train.py - data loading, LR scheduling, graph sampling, and logging of training data.

nn/models/renonet.py - contains a module of the framework shown below, as well as vmapped and serial loss functions for optimizing the loss shown below.

nn/models/models.py - contains modules for the encoder and renormalization networks (GCN, HGCN) and decoder networks (MLP, Transformer, DeepOnet).



![page1_image1](https://github.com/nngabe/msonet/assets/50005216/a947fa80-9a06-4818-8012-49a4186d2622)
