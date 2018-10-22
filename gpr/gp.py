"""
Copyright (c) 2017 Favour Mandanji Nyikosa
A module for Gaussian Process (GP) Regression
Author: Mandanji Nyikosa
Spec:
	Model: y = f(x) + \epsilon
	Prior  p(f)
	Data:  {X, y}
	Query: x_star
	Gaussian likelihood: p(y|X)
	Posterior:          p(f|X,y,x_star)
	Inference with SVD and Cholesky Decomposition
	SE, Compound SE, RQ & Matern Kernels
"""
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy as sp
import operator as op
import GPy as gpy
import time as timer
from   os.path import expanduser as eu
import sys
import matplotlib.pyplot as plt
import os
import random
######################################################################
####################### AUXILIARY FUNCTIONS ##########################
######################################################################
# simple distance measure - metric distance
def d_simple(x, y, l):
	"""
	This is a simple distance measure to be used in covariance
	functions that take the form
	$k(x, y, \sigma_f, l)=
	\sigma_f^2 \kappa(d_simple(x, y, l))$
	where
	$d_simple = |x-y|/l$
	"""
	d_temp = np.divide( (x-y), l )
	return la.norm( d_temp )
# product distance measure
def d_prod(x, y, prod_term):
	"""
	This is a simple distance measure to be used in covariance
	functions that take the form
	$k(x, y, \prod_term)=
	\kappa(d_simple(x, y, prod_term))$
	where
	$d_prod = prod_term * sum_{D}(x_i * y_i)$
	for D-dimenstional x and y
	"""
	d_temp = np.sum(x * y)
	return d_temp * prod_term
# squared exponential covariance function
def cov_se(x, y, params):
	"""
	The squared exponential covariance function is defined as:
	$ k = sigma_f * \exp( -0.5 * d_simple(x, y, l)^2$
	with the parameters
	param = [sigma_f, sigma_f, l] where
	l is D-dimentional and
	der_k = [d_sigma_n, d_sigma_f, d_l]
	"""
	# extract hyperparameters
	sigma_n = params[0:1, :]
	sigma_f = params[1:2, :]
	l = params[2:max(params.shape)+1, :]
	# calculate distance
	d = d_simple(x, y, l)
	# calculate covariance
	k = np.square(sigma_f) * np.exp( -0.5 * np.square(d) )
	# partial derivatives of covariance function
	d_sigma_n = np.zeros([1,1], float)
	if np.array_equal(x, y):
		d_sigma_n = 2 * sigma_n * np.ones([1,1], float)
	d_sigma_f = 2 * sigma_f * np.exp( -0.5 * np.square(d) )
	l_power_neg_3 = np.power(l, -3)
	l_by_k = np.dot(l_power_neg_3, k)
	d_l = np.dot( l_by_k,  np.square(d) )
	der_k = np.concatenate((d_sigma_n, d_sigma_f, d_l), axis=0)
	return k, der_k
# compound squared exponential covariance function- from Bishop (2006)
def cov_se_compound(x, y, params):
	"""
	The squared exponential covariance function is defined as:
	$ k = sigma_f * \exp( -0.5 * d_simple(x, y, l)^2 + bias
	+ prod_term * x * y $
	with the parameters
	param = [prod_param, bias_param, sigma_n, sigma_f, l] where
	l is D-dimentional and
	der_k = [d_prod_param, d_bias_param, d_sigma_f, d_sigma_n, d_sigma_f, d_l]
	"""
	# extract hyperparameters
	prod_param = params[0:1, :]
	bias_param = params[1:2, :]
	sigma_n = params[2:3, :]
	sigma_f = params[3:4, :]
	l = params[4:max(params.shape)+1, :]
	# calculate distances
	d_first = d_simple(x, y, l)
	d_second = d_prod(x, y, prod_param)
	# calculate terms in kernel
	first = np.square(sigma_f) * np.exp( -0.5 * np.square(d_first) )
	second = d_second
	third = bias_param
	k = first + second + third
	# partial derivatives of covariance function
	d_sigma_n = np.zeros([1,1], float)
	if np.array_equal(x, y):
		d_sigma_n = 2 * sigma_n * np.ones([1,1], float)
	d_prod_param = np.array([[np.sum(x * y)]], float)
	d_bias_param = np.array([[ 1 ]], float)
	d_sigmaf = 2 * sigma_f * np.exp( -0.5 * np.square(d_first) )
	d_l = np.dot( np.dot(np.power(l, -3), k),  np.square(d_first) )
	der_k = \
	np.concatenate((d_prod_param, \
		d_bias_param, d_sigma_n, d_sigmaf, d_l), axis=0)
	return k, der_k
# rational quadratic covariance function
def cov_rq(x, y, params):
	"""
	The rational quadratic covariance function is defined as:
	$k = sigma_f^2*(1+ 1/2*alpha * d_simple(x, y, l)^2)^-alpha$
	with the parameters
	param = [l, sigma_f] where
	l is D-dimentional
	"""
	alpha = params[0:1,:]
	sigma_n = params[1:2, :]
	sigma_f = params[2:3,:]
	l = params[3:max(params.shape)+1,:]
	d = d_simple(x, y, l)
	k = np.square(sigma_f) * np.power (( 1 + 1/(2 * alpha) \
		* np.square(d) ), -alpha)
	der_k = 0
	return k, der_k
# matern 3/2 covariance function
def cov_m32(x, y, params):
	"""
	The Matern 3/2 covariance function is defined as:
	$ k = sigma_f^2 * (1 + \sqrt{3} d_simple(x,y,l))
	\exp(- \sqrt{3} d_simple(x,y,l)) $
	with the parameters
	param = [l, sigma_f] where
	l is D-dimentional
	"""
	sigma_n = params[0:1,:]
	sigma_f = params[1:2,:]
	l = params[2:max(params.shape)+1,:]
	d = d_simple(x, y, l)
	k = np.square(sigma_f) * (1 + np.sqrt(3) * d) * np.exp( - np.sqrt(3) * d )
	der_k = 0
	return k, der_k
# matern 5/2 covariance function
def cov_m52(x, y, params):
	"""
	The Matern 5/2 covariance function is defined as:
	$ k = sigma_f^2 * (1 + \sqrt{5} d_simple(x,y,l)
		+ 5/3 d_simple(x,y,l)^2 ) \exp(- \sqrt{5} d_simple(x,y,l)) $
	with the parameters
	param = [l, sigma_f] where
	l is D-dimentional
	"""
	sigma_n = params[0:1,:]
	sigma_f = params[1:2,:]
	l = params[2:max(params.shape)+1,:]
	d = d_simple(x, y, l)
	k = np.square(sigma_f) * (1 + ( np.sqrt(5) * d ) + (5/3 * d) ) \
	* np.exp( - np.sqrt(5) * d )
	der_k = 0
	return k, der_k
# build covariance matrix
def calculate_k(covf, x, y, params):
    length_x = x.shape[0]
    length_y = y.shape[0]
    num_params = np.max( params.shape )
    der_k_list = []
    # create dictionary of derivative matrices
    for _i_ in range(num_params ):
    	der_k_list.append( np.zeros([length_x, length_y], float) )
    cov_k = np.zeros( [length_x, length_y], float )
    x_i_index = 0
    y_i_index = 0
    for x_i in x:
    	for y_i in y:
    		k_i, der_k_i = covf(x_i, y_i, params)
    		x_peg = np.mod(x_i_index, length_x )
    		y_peg = np.mod(y_i_index, length_y)
    		# assign covaraince
    		cov_k[ x_peg, y_peg ] = k_i
    		# assign gradient for each hyperparameter to its matrix in list
    		for hyp_index in range(num_params):
    			temp_i = der_k_list[hyp_index]
    			temp_i[x_peg,y_peg] = der_k_i[hyp_index]
    		y_i_index = y_i_index + 1
    	x_i_index = x_i_index + 1
    return cov_k, der_k_list
# SVD factorisation (courtesy of Yves-Laurent Kom Samo)
def svd_factorise(cov, max_cn=1e8):
    """
    Computes the inverse and the determinant of a covariance matrix in
    one go, using SVD.
    Returns a structure containing the following keys:
        inv: the inverse of the covariance matrix
        det: the determinant of the covariance matrix.
    """
    U, S, V = la.svd(cov)
    covI= np.dot(V.T, np.dot( np.diag( 1.0/(S) ), U.T) )
    res = {}
    res['inv'] = covI.copy()
    res['det'] = la.det(cov)
    res['u'] = U.copy()
    res['s'] = S.copy()
    res['v'] = V.copy()
    return res
# SVD factorisation (courtesy of Yves-Laurent Kom Samo)
def svd_factorise_yl(cov, max_cn=1e8):
    """
    Computes the inverse and the determinant of a covariance matrix in
    one go, using SVD.
    Returns a structure containing the following keys:
        inv: the inverse of the covariance matrix,
        L: the pseudo-cholesky factor US^0.5,
        det: the determinant of the covariance matrix.
    """
    U, S, V = la.svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > max_cn:
        nc = np.min([oc, max_cn])
        eps = np.min(S)*(oc-nc)/(nc-1.0)
    L = np.dot(U, np.diag(np.sqrt(S+eps)))
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    covInv = np.dot(V.T, np.dot( np.diag( 1.0/(S) ), U.T) )
    res = {}
    res['inv'] = covI.copy()
    res['inv_direct'] = covInv.copy()
    res['L'] = L.copy()
    res['det'] = np.prod(S+eps)
    res['log_det'] = np.sum(np.log(S+eps))
    res['LI'] = LI.copy()
    res['eigen_vals'] = S+eps
    res['u'] = U.copy()
    res['v'] = V.copy()
    return res
# do cholesky factorisation
def chol_factorise(K, y):
	"""
	This function performs Cholesky decomposition for quick inversion
	of the covariance matrix K
	"""
	L = la.cholesky(K)
	alpha = la.solve(L.T, la.solve(L, y) )
	return L, alpha
# log marginal likelihood function
def marginal_likelihood_chol(L,alpha,y):
	"""
	Calculates marginal likelihood p(y|X) using Cholesky decomposition
	This is the marginalisation over the function prior values of f
	"""
	n = y.shape[0]
	first_term = -0.5 * np.dot(y.T, alpha)
	lik =  first_term - (0.5 * n)* np.log(2 * np.pi) \
	- np.sum( np.log( np.diagonal(L) ))
	return lik
# log marginal likelihood function
def marginal_likelihood_svd(res,y):
	"""
	Calculates marginal likelihood p(y|X) using SVD
	This is the marginalisation over the function prior values of f
	"""
	n = y.shape[0]
	lik = np.dot( np.dot( y.T , res['inv'] ), y) \
	+ np.log(res['det']) + (n * np.log(2 * np.pi))
	return -0.5 * lik
# log marginal likelihood function
def marginal_likelihood_direct(K_inv,K_det,y):
	"""
	Calculates marginal likelihood p(y|X) directly (costly).
	This is the marginalisation over the function prior values of f
	"""
	n = y.shape[0]
	lik = np.dot( np.dot(y.T, K_inv), y ) + (np.log(K_det)) \
	+ (n * np.log(2 * np.pi))
	return -0.5 * lik
# adding measurement noise
def add_noise(K, noise):
	return K + (np.eye(K.shape[0]) * noise )
# log marginal likelihood objective function
def marginal_likelihood_objfunc(params, X, y, kernel):
	params = np.array([params]).T
	K, dK_dT = calculate_k(kernel, X, y, params)
	K = add_noise(K, params[0])
	res = svd_factorise(K, max_cn=1e8)
	log_lik = marginal_likelihood_svd(res,y)
	der_log_lik = marginal_likelihood_objfunc_der(res, y, params, dK_dT)
	return log_lik, der_log_lik
# log marginal likelihood objective function partial derivatives
def marginal_likelihood_objfunc_der(res, y, params, dK_dT):
	num_rows, num_cols = params.shape
	alpha = np.dot(res['inv'], y)
	log_lik_der = np.zeros( [num_rows, num_cols], float)
	for i in range(num_rows):
		dK_dT_i = dK_dT[i]
		log_lik_i = np.trace( np.dot( np.dot(alpha, alpha.T) - res['inv'], dK_dT_i) )
		log_lik_der[i:i+1,:] = log_lik_i
	return log_lik_der
# negative log marginal likelihood objective function - as obj_func
def obj_func(params, X, y, kernel):
	param_state_flag = 0      # default
	num_params = np.max( params.shape )
	if params.shape == (1, num_params):
		params = params.T
		param_state_flag = 1 # state 1
	elif params.shape == (num_params,):
		params = np.array([params]).T
		param_state_flag = 2 # state 2
	elif params.shape == (1, num_params):
		param_state_flag = 3 # state 3
	else:
		param_state_flag = 4 # state 4
		#error('the parameter vector is of the wrong form')
	K, dK_dT = calculate_k(kernel, X, y, params)
	K = add_noise(K, params[0])
	res = svd_factorise(K, max_cn=1e8)
	log_lik = -marginal_likelihood_svd(res,y)
	return log_lik
# negaive log marginal likelihood objective function partial
# derivatives - as fptime
def obj_func_prime(params, X, y, kernel):
	param_state_flag = 0      # default
	num_params = np.max( params.shape )
	if params.shape == (1, num_params):
		params = params.T
		param_state_flag = 1  # state 1
	elif params.shape == (num_params,):
		params = np.array([params]).T
		param_state_flag = 2  # state 2
	elif params.shape == (1, num_params):
		param_state_flag = 3  # state 3
	else:
		param_state_flag = 4  # state 4
		#error('the parameter vector is of the wrong form')
	K, dK_dT = calculate_k(kernel, X, y, params)
	K = add_noise(K, params[0])
	res = svd_factorise(K, max_cn=1e8)
	num_params = np.max( params.shape )
	if params.shape == (num_params, 1):
		num_rows, num_cols = params.shape
	elif params.shape == (1, num_params):
		num_rows, num_cols = params.shape
	elif params.shape == (num_params, ):
		num_rows = 1
		num_cols = num_params
	#num_rows, num_cols = params.shape - original line
	alpha = np.dot(res['inv'], y)
	log_lik_der = np.zeros( [num_rows, num_cols], float)
	for i in range(num_rows):
		dK_dT_i = dK_dT[i]
		log_lik_i = np.trace( np.dot( np.dot(alpha, alpha.T) - res['inv'], dK_dT_i) )
		log_lik_der[i:i+1,:] = -log_lik_i
	return log_lik_der.T[0]
######################################################################
########################### OPTIMISATION #############################
######################################################################
# Maximise marginal likelihood
def optimise_hyperparamters(params, X, y, kernel):
	solution = \
	sp.optimize.minimize(obj_func, params, args=(X, y, kernel), method='BFGS', jac=obj_func_prime)
	return solution
# Train GP
def train(params, X, y, kernel):
	solution = optimise_hyperparamters(params, X, y, kernel)
	return solution
######################################################################
########################### MAIN FUNCTIONS ###########################
######################################################################
# Perform GP inference
# predict with Cholesky
def predict_chol(x_star, X, y, kernel, params):
	num_rows, num_cols = x_star.shape
	K, _ = calculate_k(kernel, X, X, params)
	K = add_noise(K, params[0])
	L, alpha = chol_factorise(K, y)
	# loop per datapoint
	mean_y_star = np.zeros([num_rows, num_cols], float)
	var_y_star = np.zeros([num_rows, num_cols], float)
	for i in range(num_rows):
		k_star, _ = calculate_k(kernel, X, x_star[i:i+1,:], params)
		k_star_star, _ = \
		calculate_k(kernel, x_star[i:i+1,:], x_star[i:i+1,:], params)
		mean_y_star[i:i+1,:] = np.dot(k_star.T, alpha)
		temp_v = la.solve(L,k_star)
		var_y_star[i:i+1,:] = k_star_star - np.dot(temp_v.T, temp_v)
	log_lik = marginal_likelihood_chol(L,alpha,y)
	return mean_y_star, var_y_star, log_lik
# predict with SVD
def predict_svd(x_star, X, y, kernel, params):
	num_rows, num_cols = x_star.shape
	K, _ = calculate_k(kernel, X, X, params)
	K = add_noise(K, params[0])
	res = svd_factorise(K, max_cn=1e8)
	# loop per datapoint
	mean_y_star = np.zeros([num_rows, num_cols], float)
	var_y_star = np.zeros([num_rows, num_cols], float)
	for i in range(num_rows):
		k_star, _ = calculate_k(kernel, X, x_star[i:i+1,:], params)
		k_star_star, _ = \
		calculate_k(kernel, x_star[i:i+1,:], x_star[i:i+1,:], params)
		mean_y_star[i:i+1,:] = np.dot( np.dot(k_star.T,res['inv']), y)
		var_y_star[i:i+1,:] = \
		k_star_star - np.dot(np.dot(k_star.T, res['inv']), k_star)
	log_lik = marginal_likelihood_svd(res,y)
	return mean_y_star, var_y_star, log_lik
# predict via direct calculation
def predict_directly(x_star, X, y, kernel, params):
	num_rows, num_cols = x_star.shape
	K, _ = calculate_k(kernel, X, X, params)
	K = add_noise(K, params[0])
	K_inv = la.inv(K)
	K_det = la.det(K)
	# loop per datapoint
	mean_y_star = np.zeros([num_rows, num_cols], float)
	var_y_star = np.zeros([num_rows, num_cols], float)
	for i in range(num_rows):
		k_star, _ = calculate_k(kernel, X, x_star[i:i+1,:], params)
		k_star_star, _ = \
		calculate_k(kernel, x_star[i:i+1,:], x_star[i:i+1,:], params)
		mean_y_star[i:i+1,:] = \
		np.dot( np.dot( k_star.T, K_inv ), y )
		var_y_star[i:i+1,:] = \
		k_star_star - np.dot( np.dot( k_star.T, K_inv), k_star )
	log_lik = marginal_likelihood_direct(K_inv, K_det, y)
	return mean_y_star, var_y_star, log_lik
######################################################################
########################### TEST FUNCTIONS ###########################
######################################################################
# create 1d dummy data
def _create_1d_dummy_data(number_of_datapoints):
	random.seed(34738924)
	x = np.linspace(0,5,number_of_datapoints)
	x = np.array([x])
	y = 2*np.square(x) + ( np.exp(x) ) \
	+ np.random.randn(number_of_datapoints)
	#y = np.array([y])
	l = np.array([1.11])
	sigma_n = np.array([.53])
	sigma_f = np.array([.23])
	alpha = np.array([0.1])
	print "x = \n", x
	print "y = \n", y
	print "l = \n", l
	print "sigma_n = \n", sigma_n
	print "sigma_f = \n", sigma_f
	print "alpha = \n", alpha
	return x.T, y.T, l, sigma_n, sigma_f, alpha
# create 2d dummy data
def _create_2d_dummy_data(number_of_datapoints):
	random.seed(34738924)
	x1 = np.linspace(0,5,number_of_datapoints)
	x2 = np.linspace(-4,1,number_of_datapoints)
	x = np.array([x1,x2])
	y = 2*np.square(x) + ( np.exp(x) ) \
	+ np.random.randn(number_of_datapoints)
	y = y[0:1,:]
	l = np.array([.11])
	sigma_n = np.array([.51])
	sigma_f = np.array([.1])
	alpha = np.array([0.1])
	print "x = \n", x.T
	print "y = \n", y.T
	print "l = \n", l
	print "sigma_f = \n", sigma_f
	print "alpha = \n", alpha
	print "d_simple: PASS\n"
	return x.T, y.T, l, sigma_n, sigma_f, alpha
# test d_simple distance measure
def _test_d_simple(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	dist = d_simple(x, x, l)
	print "disp=\n", dist
	print "d_simple: PASS\n"
# test cov_se kernel
def _test_cov_se(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	print "x = \n", x
	print "y = \n", y
	print "params = \n", params
	k_xx, der_k_xx = cov_se(x, x, params)
	k_xy, der_k_xy = cov_se(x, y, params)
	k_yy, der_k_yy = cov_se(y, y, params)
	print "k_xx = \n", k_xx
	print "k_yy = \n", k_yy
	print "k_xy = \n", k_xy
	print "der_k_xx = \n", der_k_xx
	print "der_k_yy = \n", der_k_yy
	print "der_k_xy = \n", der_k_xy
	print "cov_se: PASS\n"
# test cov_se_compound kernel
def _test_cov_se_compound(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_f, sigma_f, sigma_n, sigma_f, l])
	print "params = ", params
	k_xx, der_k_xx = cov_se(x, x, params)
	k_xy, der_k_xy = cov_se(x, y, params)
	k_yy, der_k_yy = cov_se(y, y, params)
	print "k_xx = ", k_xx
	print "k_yy = ", k_xy
	print "k_xy = ", k_yy
	print "cov_se_compound: PASS\n"
# test cov_rq kernel
def _test_cov_rq(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_f, sigma_n, sigma_f, l])
	print "params = ", params
	k_xx = cov_rq(x, x, params)
	k_yy = cov_rq(y, y, params)
	k_xy = cov_rq(x, y, params)
	print "k_xx = ", k_xx
	print "k_yy = ", k_yy
	print "k_xy = ", k_xy
	print "cov_rq: PASS\n"
# test cov_m32 kernel
def _test_cov_m32(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	print "params =\n", params
	k_xx = cov_m32(x, x, params)
	k_yy = cov_m32(y, y, params)
	k_xy = cov_m32(x, y, params)
	print "k_xx = \n", k_xx
	print "k_yy = \n", k_yy
	print "k_xy = \n", k_xy
	print "cov_m32: PASS\n"
# test cov_m52 kernel
def _test_cov_m52(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	print "params = \n", params
	k_xx = cov_m52(x, x, params)
	k_yy = cov_m52(y, y, params)
	k_xy = cov_m52(x, y, params)
	print "k_xx = \n", k_xx
	print "k_yy = \n", k_yy
	print "k_xy = \n", k_xy
	print "cov_m52: PASS\n"
# test calculate_k 1 dimensional
def _test_calculate_k_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	print "params = ", params
	covf = cov_se
	print "x = \n",x
	print "y = \n", y
	print "l =", l
	print "sigma_n =", sigma_n
	print "sigma_f =", sigma_f
	print "alpha =", alpha
	k_xx, k_xx_ = calculate_k(covf, x, x, params)
	k_yy, k_yy_ = calculate_k(covf, y, y, params)
	k_xy, k_xy_ = calculate_k(covf, x, y, params)
	print "k_xx = \n", k_xx
	print "k_yy = \n", k_yy
	print "k_xy = \n", k_xy
	print "der_k_xx = \n", k_xx_
	print "der_k_yy = \n", k_yy_
	print "der_k_xy = \n", k_xy_
	print "calculate_1d_k: PASS\n"
	#print k_xx
# test calculate_k 2 dimensional
def _test_calculate_k_2d(number_of_datapoints, ):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l, l+0.45])
	#params = params.T
	print "params = ", params
	covf  = cov_se
	print "x = \n",x
	print "y = \n", y
	print "l = \n", l
	print "sigma_n =", sigma_n
	print "sigma_f =", sigma_f
	print "alpha = ", alpha
	k_xx, k_xx_ = calculate_k(covf, x, x, params)
	k_yy, k_yy_ = calculate_k(covf, y, y, params)
	k_xy, k_xy_ = calculate_k(covf, x, y, params)
	print "k_xx = \n", k_xx
	print "k_yy = \n", k_yy
	print "k_xy = \n", k_xy
	print "der_k_xx = \n", k_xx_
	print "der_k_yy = \n", k_yy_
	print "der_k_xy = \n", k_xy_
	print "calculate_2d_k: PASS\n"
# test solve SVD - 1d
def _test_svd_factorise_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	res = svd_factorise(cov, max_cn=1e8)
	print "inv = \n", res['inv']
	print "normal_inv = \n", la.inv(cov)
	print "normal_L = \n", la.cholesky(cov)
	print "det = \n", res['det']
	print "LI = \n", res['LI'], "\n"
	print "log_det = \n", res['log_det']
	print "eigen_vals = \n", res['eigen_vals']
	print "normal_eigs = \n", la.eig(cov)
	print "u = \n", res['u']
	print "v = \n", res['v']
	print "svd_factorise_1d: PASS\n"
# test solve SVD - 2d
def _test_svd_factorise_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	res = svd_factorise_yl(cov, max_cn=1e8)
	print "inv = \n", res['inv']
	print "normal_inv = \n", la.inv(cov)
	print "normal_L = \n", la.cholesky(cov)
	print "det = \n", res['det']
	print "LI = \n", res['LI'], "\n"
	print "log_det = \n", res['log_det']
	print "eigen_vals = \n", res['eigen_vals']
	print "normal_eigs = \n", la.eig(cov)
	print "u = \n", res['u']
	print "v = \n", res['v']
	print "svd_factorise_2d: PASS"
# test cholesky factorisation - 1d
def _test_chol_factorise_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	L, alpha = chol_factorise(cov, y)
	print "L = \n", L
	print "alpha = \n", alpha
	print "_test_chol_factorise_1d: PASS"
# test cholesky factorisation - 2d
def _test_chol_factorise_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	L, alpha = chol_factorise(cov, y)
	print "L = \n", L
	print "alpha = \n", alpha
	print "_test_chol_factorise_2d: PASS"
# test marginal likelihood done via cholesky decomposition - 1d
def _test_marginal_likelihood_chol_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	L, alpha = chol_factorise(cov, y)
	print "L = \n", L
	print "alpha =\n", alpha
	print "y =\n", y
	print "x =\n", x
	ml = marginal_likelihood_chol(L,alpha,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_1d: PASS"
# test marginal likelihood done via cholesky decomposition - 2d
def _test_marginal_likelihood_chol_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	L, alpha = chol_factorise(cov, y)
	print "L = \n", L
	print "alpha =\n", alpha
	print "y =\n", y
	print "x =\n", x
	ml = marginal_likelihood_chol(L,alpha,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_2d: PASS"
# test marginal likelihood done via SVD - 1d
def _test_marginal_likelihood_svd_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	res = svd_factorise(cov, max_cn=1e8)
	print "y =\n", y
	print "x =\n", x
	ml = marginal_likelihood_svd(res,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_1d: PASS"
# test marginal likelihood done via SVD - 2d
def _test_marginal_likelihood_svd_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	res = svd_factorise(cov, max_cn=1e8)
	print "y =\n", y
	print "x =\n", x
	ml = marginal_likelihood_svd(res,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_2d: PASS"
# test marginal likelihood done via Direct - 1d
def _test_marginal_likelihood_direct_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	cov_inv = la.inv(cov)
	cov_det = la.det(cov)
	print "y =\n", y
	print "x =\n", x
	print "cov_inv =\n", y
	print "cov_det =\n", x
	ml = marginal_likelihood_direct(cov_inv,cov_det,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_1d: PASS"
# test marginal likelihood done via Direct - 2d
def _test_marginal_likelihood_direct_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	cov_inv = la.inv(cov)
	cov_det = la.det(cov)
	print "y =\n", y
	print "x =\n", x
	print "cov_inv =\n", y
	print "cov_det =\n", x
	ml = marginal_likelihood_direct(cov_inv,cov_det,y)
	print "marginal likelihood = \n", ml
	print "_test_marginal_likelihood_chol_2d: PASS"
# test adding noise
def _test_add_noise(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	cov, _ = calculate_k(covf, x, x, params)
	noise = 0.002
	print "K = \n", cov
	print "der_K = \n", _
	print "K_noisy = \n", add_noise(cov, noise)
	print "_test_add_noise: PASS"
# test predictions via chol
def _test_predict_chol(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	x_star = x + 2.13
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	mean_y_star, var_y_star, log_lik = \
	predict_chol(x_star, x, y, covf, params)
	print "\nPredict using Chol"
	print "y =\n", y
	print "x =\n", x
	print "x_star =\n", x_star
	print "mean = \n", mean_y_star
	print "var = \n", var_y_star
	print "log_lik = \n", log_lik
	print "_test_predict_chol: PASS"
# test predictions via svd
def _test_predict_svd(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	x_star = x + 2.13
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	mean_y_star, var_y_star, log_lik = \
	predict_svd(x_star, x, y, covf, params)
	print "\nPredict using SVD"
	print "y =\n", y
	print "x =\n", x
	print "x_star =\n", x_star
	print "mean = \n", mean_y_star
	print "var = \n", var_y_star
	print "log_lik = \n", log_lik
	print "_test_predict_svd: PASS"
# test direct prediction
def _test_predict_directly(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	x_star = x + 2.13
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	mean_y_star, var_y_star, log_lik = \
	predict_directly(x_star, x, y, covf, params)
	print "\nPredict directly"
	print "y =\n", y
	print "x =\n", x
	print "x_star =\n", x_star
	print "mean = \n", mean_y_star
	print "var = \n", var_y_star
	print "log_lik = \n", log_lik
	print "_test_predict_directly: PASS"
# test the objective function for maximum marginal likelihood
def _test_marginal_likelihood_objfunc(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	covf = cov_se
	log_lik = obj_func(params, x, y, covf)
	print "\nLog Lik Obj Func"
	print "x =\n", x
	print "y =\n", y
	print "params =\n", params
	print "log_lik = \n", log_lik
	print "_test_marginal_likelihood_objfunc: PASS"
# test maximum marginal likelihood optimization
def _test_marginal_likelihood_optimisation(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	params = np.array([sigma_n, sigma_f, l])
	params = [sigma_n, sigma_f, l]
	print "bf 1: params =\n", params
	covf = cov_se
	solution = optimise_hyperparamters(params, x, y, covf)
	print "\nLog Lik Obj Func"
	print "x =\n", x
	print "y =\n", y
	print "params =\n", params
	print "solution = \n", solution
	print "_test_marginal_likelihood_optimisation: PASS"
# test full pipeline
def _test_pipeline_1d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	x_star = x + 2.13
	params = [sigma_n, sigma_f, l]
	covf = cov_se
	solution = optimise_hyperparamters(params, x, y, covf)
	opt_params = np.array([solution.x]).T
	mean_y_star, var_y_star, log_lik = \
	predict_svd(x_star, x, y, covf, opt_params)
	print "\nLog Lik Obj Func"
	print "x =\n", x
	print "y =\n", y
	print "params =\n", params
	print "solution = \n", solution
	print "mean_y_star = \n", mean_y_star
	print "var_y_star = \n", var_y_star
	print "log_lik = \n", log_lik
	print "_test_pipeline_1d: PASS"
# test full pipeline
def _test_pipeline_2d(number_of_datapoints):
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	x_star = x + 2.13
	params = [sigma_n, sigma_f, l, l]
	covf = cov_se
	solution = optimise_hyperparamters(params, x, y, covf)
	opt_params = np.array([solution.x]).T
	mean_y_star, var_y_star, log_lik = \
	predict_svd(x_star, x, y, covf, opt_params)
	print "\nLog Lik Obj Func"
	print "x =\n", x
	print "y =\n", y
	print "params =\n", params
	print "solution = \n", solution
	print "mean_y_star = \n", mean_y_star
	print "var_y_star = \n", var_y_star
	print "log_lik = \n", log_lik
	print "_test_pipeline_2d: PASS"
######################################################################
############################## TESTS  ################################
######################################################################
# run tests
def run_tests():

	number_of_datapoints = 3

	print "#####################"
	print "\nCREATE 1D DUMMY DATA"
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_1d_dummy_data(number_of_datapoints)
	print "#####################"

	print "\nCREATE 2D DUMMY DATA"
	x, y, l, sigma_n, sigma_f, alpha = \
	_create_2d_dummy_data(number_of_datapoints)
	print "#####################"

	print "\nD_SIMPLE"
	_test_d_simple(number_of_datapoints)
	print "#####################"

	print "\nCOV_SE"
	_test_cov_se(number_of_datapoints)
	print "#####################"

	print "\nCOV_SE_COMPOUND"
	_test_cov_se_compound(number_of_datapoints)
	print "#####################"

	print "\nCOV_RQ"
	_test_cov_rq(number_of_datapoints)
	print "#####################"

	print "\nCOV_M32"
	_test_cov_m32(number_of_datapoints)
	print "#####################"

	print "\nCOV_M52"
	_test_cov_m52(number_of_datapoints)
	print "#####################"

	print "\nBUILD_COVARIANCE_MATRIX K"
	_test_calculate_k_1d(number_of_datapoints)
	_test_calculate_k_2d(number_of_datapoints)
	print "#####################"

	print "\nSVD_Factorise"
	_test_svd_factorise_1d(number_of_datapoints)
	_test_svd_factorise_2d(number_of_datapoints)
	print "#####################"

	print "\nCHOL_FACTORISE"
	_test_chol_factorise_1d(number_of_datapoints)
	_test_chol_factorise_2d(number_of_datapoints)
	print "#####################"

	print "\nMARGINAL_LIKELIHOOD_CHOL"
	_test_marginal_likelihood_chol_1d(number_of_datapoints)
	_test_marginal_likelihood_chol_2d(number_of_datapoints)
	print "#####################"

	print "\nMARGINAL_LIKELIHOOD_SVD"
	_test_marginal_likelihood_svd_1d(number_of_datapoints)
	_test_marginal_likelihood_svd_2d(number_of_datapoints)
	print "#####################"

	print "\nMARGINAL_LIKELIHOOD_DIRECT"
	_test_marginal_likelihood_direct_1d(number_of_datapoints)
	_test_marginal_likelihood_direct_2d(number_of_datapoints)
	print "#####################"

	print "\nTEST_ADD_NOISE"
	_test_add_noise(number_of_datapoints)
	print "#####################"

	print "\nTEST_PREDICTIONS"
	_test_predict_chol(number_of_datapoints)
	_test_predict_svd(number_of_datapoints)
	_test_predict_directly(number_of_datapoints)
	print "#####################"

	print "\nTEST HYP OBJ FUNCTION"
	_test_marginal_likelihood_objfunc(number_of_datapoints)
	print "#####################"

	print "\nHYP OPTIMIZATION"
	_test_marginal_likelihood_objfunc(number_of_datapoints)
	print "#####################"

	print "\nTEST HYP OBJ FUNCTION"
	_test_marginal_likelihood_optimisation(number_of_datapoints)
	print "#####################"

	print "\nTEST 1D PIPELINE"
	_test_pipeline_1d(number_of_datapoints)
	print "#####################"

	print "\nTEST 2D PIPELINE"
	_test_pipeline_2d(number_of_datapoints)
	print "#####################"

######################################################################
########################### RUN CODE  ################################
######################################################################

# run_tests()

######################################################################
######################################################################
########################### CODE GRAVEYARD ###########################
######################################################################
######################################################################
# SVD factorisation (courtesy of Yves-Laurent Kom Samo)
# def svd_factorise_yl(cov, max_cn=1e8):
#     """
#     Computes the inverse and the determinant of a covariance matrix in
#     one go, using SVD.
#     Returns a structure containing the following keys:
#         inv: the inverse of the covariance matrix,
#         L: the pseudo-cholesky factor US^0.5,
#         det: the determinant of the covariance matrix.
#     """
#     U, S, V = la.svd(cov)
#     eps = 0.0
#     oc = np.max(S)/np.min(S)
#     if oc > max_cn:
#         nc = np.min([oc, max_cn])
#         eps = np.min(S)*(oc-nc)/(nc-1.0)
#     L = np.dot(U, np.diag(np.sqrt(S+eps)))
#     LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
#     covI= np.dot(LI.T, LI)
#     covInv = np.dot( V, np.dot(1.0/(S), U.T) )
#     res = {}
#     res['inv'] = covI.copy()
#     res['inv_'] = covInv.copy()
#     res['L'] = L.copy()
#     res['det'] = np.prod(S+eps)
#     res['log_det'] = np.sum(np.log(S+eps))
#     res['LI'] = LI.copy()
#     res['eigen_vals'] = S+eps
#     res['u'] = U.copy()
#     res['v'] = V.copy()
#     return res
######################################################################
