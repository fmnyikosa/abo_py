
import numpy as np
from   gpflowopt.domain import ContinuousParameter


def fx(X):
    X  = np.atleast_2d(X)
    return np.sum(np.square(X), axis=1)[:, None]

domain = ContinuousParameter('x1', -2, 2) + ContinuousParameter('x2', -1, 2)
domain

######################################################################################


import GPflow                as  gpflow
from   gpflowopt.bo          import BayesianOptimizer
from   gpflowopt.design      import LatinHyperCube
from   gpflowopt.acquisition import ExpectedImprovement
from   gpflowopt.optim       import SciPyOptimizer

# Use standard Gaussian process Regression
lhd       = LatinHyperCube(21, domain)
X         = lhd.generate()
Y         = fx(X)
model     = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(2, ARD=True))
model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)

# Now create the Bayesian Optimizer
alpha     = ExpectedImprovement(model)
optimizer = BayesianOptimizer(domain, alpha)

# Run the Bayesian optimization
with optimizer.silent():
    r     = optimizer.optimize(fx, n_iter=15)

print(r)