"""
Copyright (c) 2017 Favour Mandanji Nyikosa
A demo script for Gaussian process regression.
Author: Mandanji Nyikosa
Spec:
	Model: y = f(x) + \epsilon
	Prior p(f)
	Data: {X, y}
	Query: x_star
	Gaussian likelihood p(y|X)
	Posterior: p(f|X,y,x_star)
	Inference with SVD and Cholesky Decomposition
	SE, RQ & Matern Kernels

"""

# import stuff
import gpflow as GPflow
import numpy as np
import linalg

# get sythetic training and test data
xtr =
ytr =
xte =

# define GP model
