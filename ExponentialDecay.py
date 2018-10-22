from   GPy.kern import Kern
import numpy as np

class ExponentialDecay(Kern):

	def __init__(self,input_dim,variance=1.,lengthscale=1.,power=1.,active_dims=None):
	    super(ExponentialDecay, self).__init__(input_dim, active_dims, 'exponential_decay')
	    assert input_dim == 1, "For this kernel we assume input_dim=1"
	    self.variance    = Param('variance', variance)
	    self.lengthscale = Param('lengtscale', lengthscale)
	    self.power       = Param('power', power)
	    self.add_parameters(self.variance, self.lengthscale, self.power)

	def K(self,X,X2):
	    if X2 is None: 
	    	X2 = X
	    dist2  = np.square((X-X2.T)/self.lengthscale)
	    return self.variance*(1 + dist2/2.)**(-self.power)

	def Kdiag(self,X):
	    return self.variance*np.ones(X.shape[0])

	def update_gradients_full(self, dL_dK, X, X2):
	    if X2 is None: 
	    	X2 = X

	    dist2 = np.square((X-X2.T)/self.lengthscale)

	    dvar = (1 + dist2/2.)**(-self.power)
	    dl   = self.power * self.variance * dist2 * self.lengthscale**(-3) * (1 + dist2/2./self.power)**(-self.power-1)
	    dp   = - self.variance * np.log(1 + dist2/2.) * (1 + dist2/2.)**(-self.power)

	    self.variance.gradient = np.sum(dvar*dL_dK)
	    self.lengthscale.gradient = np.sum(dl*dL_dK)
	    self.power.gradient = np.sum(dp*dL_dK)


	def update_gradients_diag(self, dL_dKdiag, X):
	    self.variance.gradient = np.sum(dL_dKdiag)
	    # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged    

	def gradients_X(self,dL_dK,X,X2):
    """derivative of the covariance matrix with respect to X."""
	    if X2 is None: 
	    	X2 = X
	    dist2  = np.square((X-X2.T)/self.lengthscale)

	    dX = -self.variance*self.power * (X-X2.T)/self.lengthscale**2 *  (1 + dist2/2./self.lengthscale)**(-self.power-1)
	    return np.sum(dL_dK*dX,1)[:,None]


	def gradients_X_diag(self,dL_dKdiag,X):
	    # no diagonal gradients
	    pass    























