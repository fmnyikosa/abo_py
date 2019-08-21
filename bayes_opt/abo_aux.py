"""
An auxillary module for Adaptive Bayesian Optimisation (ABO)

Author: Mandanji Nyikosa

Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com>  19th/March/2018
"""

import GPy        as gpy
import numpy      as np
import scipy.io   as sio


# ------------ Auxillary functions -----------------

# --- softmax models

def get_softmax_adam_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-softmax-adam-mnist-21-March-2018-17-27-23-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=31941,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=3.53,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.6121, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model


def get_softmax_nest_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-softmax-nesterov-mnist-21-March-2018-16-19-56-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=7758.8,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.691,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.36767, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-8
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_softmax_sgd_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-softmax-sgd-mnist-21-March-2018-18-58-00-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=3095.368,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.5,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=1.07, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

# --- multi 1 models

def get_multi1_adam_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer1-adam-mnist-21-March-2018-17-39-34-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=750.954,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.7158755,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.10371, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-127
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model	

def get_multi1_nest_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer1-nesterov-mnist-21-March-2018-16-24-54-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=1722.9656452,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.383,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.183967, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_multi1_sgd_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer1-sgd-mnist-21-March-2018-19-02-40-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=577.4644,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.327896,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.7987, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

# --- multi2 models

def get_multi2_adam_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer2-adam-mnist-21-March-2018-18-11-05-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=39,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=4.2,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.17399, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	gp_model.likelihood.variance.constrain_bounded(1e-800, 1.7)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_multi2_nest_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer2-nesterov-mnist-21-March-2018-16-32-46-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=404.3,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.152,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.127, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-100
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.1)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model


def get_multi2_sgd_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-multilayer2-sgd-mnist-21-March-2018-19-09-56-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=240,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.215,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=1.00, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-67
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

# --- conv models

def get_conv_adam_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-convolutional-adam-cifar10-21-March-2018-17-24-38-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=2632783.684,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=5.41575,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=163.577, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_conv_nest_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-convolutional-nesterov-cifar10-21-March-2018-16-17-16-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=28354685.4521,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=4.5,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=268.5627, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-132
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_conv_sgd_model():
	data  = sio.loadmat(
		'./logs/bench1/bench-convolutional-sgd-cifar10-21-March-2018-18-55-19-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=855470,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=3.95,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=34, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 1e-8
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model	


# ---------------- Main Function ------------

#  checks models and optimizers and gets initial model
def get_gp_model( model_ , optimizer_ ):

	if   model_   == 'softmax'       and optimizer_ == 'adam':
		gp_model  = get_softmax_adam_model()
	elif model_   == 'softmax'       and optimizer_ == 'nesterov':
		gp_model  = get_softmax_nest_model()
	elif model_   == 'softmax'       and optimizer_ == 'sgd':
		gp_model  = get_softmax_sgd_model()
	elif model_   == 'multilayer1'   and optimizer_ == 'adam':
		gp_model  = get_multi1_adam_model() 
	elif model_   == 'multilayer1'   and optimizer_ == 'nesterov':
		gp_model  = get_multi1_nest_model()
	elif model_   == 'multilayer1'   and optimizer_ == 'sgd':
		gp_model  = get_multi1_sgd_model()
	elif model_   == 'multilayer2'   and optimizer_ == 'adam':
		gp_model  = get_multi2_adam_model()
	elif model_   == 'multilayer2'   and optimizer_ == 'nesterov':
		gp_model  = get_multi2_nest_model()
	elif model_   == 'multilayer2'   and optimizer_ == 'sgd':
		gp_model  = get_multi2_sgd_model()
	elif model_   == 'convolutional' and optimizer_ == 'adam':
		gp_model  = get_conv_adam_model()
	elif model_   == 'convolutional' and optimizer_ == 'nesterov':
		gp_model  = get_conv_nest_model()
	elif model_   == 'convolutional' and optimizer_ == 'sgd':
		gp_model  = get_conv_sgd_model()

	return gp_model
