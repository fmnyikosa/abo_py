"""
An auxillary module for Adaptive Bayesian Optimisation (ABO) - Validation

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
		'./logs/bench_v0/b-v-softmax-adam-mnist-19-March-2018-05-37-52-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=56048715,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=12.275,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=2929.85, lengthscale=n_epochs,
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


def get_softmax_nest_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-softmax-nesterov-mnist-19-March-2018-05-26-08-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=1943,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.694,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.325, lengthscale=n_epochs,
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
		'./logs/bench_v0/b-v-softmax-sgd-mnist-19-March-2018-05-50-20-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=1248,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.5,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.887, lengthscale=n_epochs,
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

# --- multi1 models

def get_multi1_adam_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer1-adam-mnist-19-March-2018-05-38-59-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=1.16,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=473,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=7, lengthscale=n_epochs,
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

def get_multi1_nest_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer1-nesterov-mnist-19-March-2018-05-26-59-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=561,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.43,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.246, lengthscale=n_epochs,
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

def get_multi1_sgd_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer1-sgd-mnist-19-March-2018-05-51-07-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=384,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.42,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.82, lengthscale=n_epochs,
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

# --- multi2 models

def get_multi2_adam_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer2-adam-mnist-19-March-2018-05-40-58-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=0.401,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=51.333,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=97.067, lengthscale=n_epochs,
	                active_dims=[1], ARD=False )
	k           = k_t_1 * k_t_2 * k_s
	# define model
	gp_model    = gpy.models.GPRegression( x, y, k )
	# set likelihood variance
	gp_model.likelihood.variance                = 0.0592
	gp_model.likelihood.variance.constrain_bounded(1e-800, 0.0592)
	# fix some hyperparamters
	gp_model.mul.Exponential.variance.constrain_fixed()
	gp_model.mul.rbf.variance.constrain_fixed()
	gp_model.mul.rbf_1.lengthscale.constrain_fixed()

	return gp_model

def get_multi2_nest_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer2-nesterov-mnist-19-March-2018-05-28-20-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=316,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.23,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=0.288, lengthscale=n_epochs,
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


def get_multi2_sgd_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-multilayer2-sgd-mnist-19-March-2018-05-52-18-PM-25-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(25)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 25
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=351,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=2.3,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=1.4, lengthscale=n_epochs,
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

# --- conv models

def get_conv_adam_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-convolutional-adam-cifar10-19-March-2018-05-37-24-PM-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=1568569,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=4.24,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=43, lengthscale=n_epochs,
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

def get_conv_nest_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-convolutional-nesterov-cifar10-19-March-2018-05-25-40-PM-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=200096333,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=7.75,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=5575, lengthscale=n_epochs,
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

def get_conv_sgd_model():
	data  = sio.loadmat(
		'./logs/bench_v0/b-v-convolutional-sgd-cifar10-19-March-2018-05-49-54-PM-100-epochs.mat')
	eta_temp    = data['eta_epochs'].T
	epochs_temp = np.arange(100)[:,None] 
	x           = np.hstack( (epochs_temp , eta_temp) )
	y           = data['loss_epochs'].T
	# define kernel
	n_epochs    = 100
	k_t_1       = gpy.kern.Exponential( input_dim=1,
	                variance=1.0, lengthscale=2357644,
	                active_dims=[0], ARD=False )
	k_t_2       = gpy.kern.RBF(     input_dim=1,
	                variance=1.0, lengthscale=4,
	                active_dims=[0], ARD=False )
	k_s         = gpy.kern.RBF(     input_dim=1,
	                variance=61, lengthscale=n_epochs,
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
