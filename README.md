# Adaptive Learning Rate Tuning in PyTorch using Adaptive Bayesian Optimization (ABO)

This repo contains code that implements the following:

- An implementation of Adaptive Bayesian Optimization (ABO) using Sheffield Machine Learning's GPy and GPyOpt frameworks.
- A modified `torch.optim`  the package that includes ABO implemented for the adaptive tuning of the learning rate (per epoch of training).
- A set of experiments to test the efficacy of the adaptive learning rate tunning procedure.
