"""
A module for Adaptive Bayesian Optimisation (ABO)
Author: Mandanji Nyikosa

Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com>  6th/May/2015
"""

import cma
import sys
import os
import scipy                as sp
import numpy                as np
import time                 as timer
import operator             as op
import gpflow               as gpflow
import GPy                  as gpy

import numpy.linalg         as la
import matplotlib.pyplot    as plt
import scipy.io             as sio

from   scipy.stats          import norm
from   scipy.optimize       import differential_evolution
# from   DIRECT             import solve
from   pyswarm              import pso
from   os.path              import expanduser as eu
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib import cm

######################################################################################
#                            ACQUISITION FUNCTIONS
#-------------------------------------------------------------------------------------

def acquisition_ei( x0 , gp_model ):
    """
    Expected Improvement acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    ei = acquisition_ei( x0, gp_model )

    where

        x0:             Sample position [N X DIM] numpy vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Expected impprovement (NB: The higher, the better).
                        We return negative version for minimization so
                        the lower the better
    """

    x0           = x0[:,None].T # add extra axis

    mean_, var_  = gp_model.predict_y( x0 )

    threshold    = np.min( gp_model.Y.value )

    sDev         = np.sqrt( var_ )

    if sDev > 0:
        Z        = ( threshold - mean_  ) / sDev
        CDF      = norm.cdf( Z )
        PDF      = norm.pdf( Z )
        ei       = sDev *  (  (Z * CDF) + PDF )
        ei       = - ei
    else:
        ei       = 0

    return ei

def acquisition_ei_with_derivatives( x0 , gp_model ):
    """
    Expected Improvement acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    ei = acquisition_ei( x0, gp_model )

    where

        x0:             Sample position [N X DIM] numpy vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Expected impprovement (NB: The higher, the better).
                        We return negative version for minimization so
                        the lower the better
    """

    x0           = x0[:,None].T # add extra axis
    x0_          = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_, var_  = gp_model.predict_y( x0_ )

    threshold    = np.min( gp_model.Y.value )

    sDev         = np.sqrt( var_ )

    if sDev > 0:
        Z        = ( threshold - mean_  ) / sDev
        CDF      = norm.cdf( Z )
        PDF      = norm.pdf( Z )
        ei       = sDev *  (  (Z * CDF) + PDF )
        ei       = - ei
    else:
        ei       = 0

    return ei

def acquisition_ei_abo( x0 , gp_model  ):
    """
    Expected Improvement acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    ei  = acquisition_ei_abo( x0, gp_model )

    where

        x0:             Sample position [N X DIM] numpy vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Expected impprovement (NB: The higher, the better).
                        We return negative version for minimization so
                        the lower the better

    """

    x0           = x0[:,None].T # add extra axis

    mean_, var_  = gp_model.predict_y( x0 )

    threshold    = min( gp_model.Y.value[ -6 : -1, : ] )

    sDev         = np.sqrt( var_ )

    if sDev > 0:
        Z        = ( threshold - mean_  ) / sDev
        CDF      = norm.cdf( Z )
        PDF      = norm.pdf( Z )
        ei       = sDev *  (  (Z * CDF) + PDF )
        ei       = - ei
    else:
        ei       = 0

    return ei

def acquisition_ei_abo_with_derivatives( x0 , gp_model  ):
    """
    Expected Improvement acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    ei  = acquisition_ei_abo_with_derivatives( x0, gp_model )

    where

        x0:             Sample position [N X DIM] numpy vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Expected impprovement (NB: The higher, the better).
                        We return negative version for minimization so
                        the lower the better

    """

    x0           = x0[:,None].T # add extra axis
    x0_          = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_, var_  = gp_model.predict_y( x0_ )

    threshold    = min( gp_model.Y.value[ -6 : -1, : ] )

    sDev         = np.sqrt( var_ )

    if sDev > 0:
        Z        = ( threshold - mean_  ) / sDev
        CDF      = norm.cdf( Z )
        PDF      = norm.pdf( Z )
        ei       = sDev *  (  (Z * CDF) + PDF )
        ei       = - ei
    else:
        ei       = 0

    return ei

def acquisition_el( x0, gp_model ):
    """
    Expected Loss acquistion function for Bayesian optimization from
    Osborne et. al (2011) [Gaussian Processes for Global Optimization].
    It is based on the minimization formalisation of the global optimisation
    problem,

                           MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function
    This is unlike the expected improvement which you would need to maximize
    to obtain the optimal sample position.

    Usage:

    el    = acquisition_el( x0, gp_model )

    where

       x0:             Sample position [N X DIM] vector or matrix
       gp_model:       Pre-trained GPflow Gaussian process model object
       el:             Expected loss (NB: The lower, the better)
    """

    x0            = x0[:,None].T # add extra axis

    mean_,var_    = gp_model.predict_y( x0 )

    threshold     = np.min( gp_model.Y.value )

    sDev          = np.sqrt( var_)
    CDF           = norm.cdf( threshold, mean_ , sDev )
    PDF           = norm.pdf( threshold, mean_ , sDev )
    el            = threshold + ( (mean_ - threshold) * CDF ) - ( sDev * PDF)

    return        el

def acquisition_el_with_derivatives( x0, gp_model ):
    """
    Expected Loss acquistion function for Bayesian optimization from
    Osborne et. al (2011) [Gaussian Processes for Global Optimization].
    It is based on the minimization formalisation of the global optimisation
    problem,

                           MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function
    This is unlike the expected improvement which you would need to maximize
    to obtain the optimal sample position.

    Usage:

    el    = acquisition_el_with_derivatives( x0, gp_model )

    where

       x0:             Sample position [N X DIM] vector or matrix
       gp_model:       Pre-trained GPflow Gaussian process model object
       el:             Expected loss (NB: The lower, the better)
    """

    x0            = x0[:,None].T # add extra axis
    x0_           = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_,var_    = gp_model.predict_y( x0_ )

    threshold     = np.min( gp_model.Y.value )

    sDev          = np.sqrt( var_)
    CDF           = norm.cdf( threshold, mean_ , sDev )
    PDF           = norm.pdf( threshold, mean_ , sDev )
    el            = threshold + ( (mean_ - threshold) * CDF ) - ( sDev * PDF)

    return        el

def acquisition_el_abo( x0, gp_model ):
    """
    Expected Loss acquistion function for Bayesian optimization from
    Osborne et. al (2011) [Gaussian Processes for Global Optimization].
    It is based on the minimization formalisation of the global optimisation
    problem,

                           MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function
    This is unlike the expected improvement which you would need to maximize
    to obtain the optimal sample position.

    Usage:

    el    = acquisition_el( x0, gp_model )

    where

       x0:             Sample position [N X DIM] vector or matrix
       gp_model:       Pre-trained GPflow Gaussian process model object
       el:             Expected loss (NB: The lower, the better)
    """

    x0            = x0[:,None].T # add extra axis

    mean_,var_    = gp_model.predict_y( x0 )


    threshold     = min( gp_model.Y.value[ -6 : -1, : ] )

    sDev          = np.sqrt( var_)
    CDF           = norm.cdf( threshold, mean_ , sDev )
    PDF           = norm.pdf( threshold, mean_ , sDev )
    el            = threshold + ( (mean_ - threshold) * CDF ) - ( sDev * PDF)

    return        el

def acquisition_el_abo_with_derivatives( x0, gp_model ):
    """
    Expected Loss acquistion function for Bayesian optimization from
    Osborne et. al (2011) [Gaussian Processes for Global Optimization].
    It is based on the minimization formalisation of the global optimisation
    problem,

                           MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function
    This is unlike the expected improvement which you would need to maximize
    to obtain the optimal sample position.

    Usage:

    el    = acquisition_el_abo_with_derivatives( x0, gp_model )

    where

       x0:             Sample position [N X DIM] vector or matrix
       gp_model:       Pre-trained GPflow Gaussian process model object
       el:             Expected loss (NB: The lower, the better)
    """

    x0            = x0[:,None].T # add extra axis
    x0_           = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_,var_    = gp_model.predict_y( x0_ )


    threshold     = min( gp_model.Y.value[ -6 : -1, : ] )

    sDev          = np.sqrt( var_)
    CDF           = norm.cdf( threshold, mean_ , sDev )
    PDF           = norm.pdf( threshold, mean_ , sDev )
    el            = threshold + ( (mean_ - threshold) * CDF ) - ( sDev * PDF)

    return        el


def calculateUCBKappa( past_iterations, dimensionality, delta ):

    top        = ( past_iterations**((dimensionality / 2) + 2) ) * (np.pi**2)
    bottom     = 3 * delta
    kappa      = np.sqrt( 2 * np.log( top / bottom ) )

    return     kappa


def acquisition_lcb( x0 , gp_model, metadata ):
    """
    Lower Confidence Bound acquistion function for Bayesian optimization from
    Srinivas et. al (2010) [Gaussian Processes for Global Optimization].
    It is based on the maximization formalisation of the global optimisation
    problem,

                               MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function

    Usage:

    [ucb, g, post_metadata] = acquisitionLCB( x0, gp_model, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        metadata:       Struct of metadata from a GP training
        ucb:            Upper confidence bounds (NB: The higher, the better)
    """
    x0               = x0[:,None].T # add extra axis
    mean_ , var_     = gp_model.predict_y( x0 )

    iterations       = metadata['iterations']
    dimensionality   = metadata['dimensionality']
    delta            = metadata['delta']
    kappa            = calculateUCBKappa( iterations, dimensionality, delta )
    practical_factor = (1/5)  #  from Srinivas et al. (2010)
    kappa            = kappa * practical_factor

    sDev             = np.sqrt( var_ )
    lcb              = mean_ - (kappa * sDev)
    lcb              = lcb

    return           lcb

def acquisition_lcb_with_derivatives( x0 , gp_model, metadata ):
    """
    Lower Confidence Bound acquistion function for Bayesian optimization from
    Srinivas et. al (2010) [Gaussian Processes for Global Optimization].
    It is based on the maximization formalisation of the global optimisation
    problem,

                               MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function

    Usage:

    [ucb, g, post_metadata] = acquisition_lcb_with_derivatives( x0, gp_model, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        metadata:       Struct of metadata from a GP training
        ucb:            Upper confidence bounds (NB: The higher, the better)
    """
    x0               = x0[:,None].T # add extra axis
    x0_              = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_ , var_     = gp_model.predict_y( x0_ )

    iterations       = metadata['iterations']
    dimensionality   = metadata['dimensionality']
    delta            = metadata['delta']
    kappa            = calculateUCBKappa( iterations, dimensionality, delta )
    practical_factor = (1/5)  #  from Srinivas et al. (2010)
    kappa            = kappa * practical_factor

    sDev             = np.sqrt( var_ )
    lcb              = mean_ - (kappa * sDev)
    lcb              = lcb

    return           lcb

def acquisition_mm( x0, gp_model ):
    """
    Minimum Mean acquistion function for Bayesian optimizatio.
    It is based on the minimization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MINIMUM this acquisition function

    Usage:

    [mean_, g, post_metadata] = acquisitionMM(x0)

    where

        x0:             Sample position [N X DIM] vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Minimum Mean (NB: The lower, the better)

    """
    x0           = x0[:,None].T # add extra axis
    mean_ , var_ = gp_model.predict_y( x0 )
    # print x0
    # print mean_
    return       mean_

def acquisition_mm_with_derivatives( x0, gp_model ):
    """
    Minimum Mean acquistion function for Bayesian optimizatio.
    It is based on the minimization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MINIMUM this acquisition function

    Usage:

    [mean_, g, post_metadata] = acquisitionMM(x0)

    where

        x0:             Sample position [N X DIM] vector or matrix
        gp_model:       Pre-trained GPflow Gaussian process model object
        ei:             Minimum Mean (NB: The lower, the better)

    """
    x0           = x0[:,None].T # add extra axis
    x0_          = np.hstack( ( x0 , np.zeros_like( x0[:,0:1] ) )  )
    mean_ , var_ = gp_model.predict_y( x0_ )
    # print x0
    # print mean_
    return       mean_

######################################################################################
#                           ACQUISITION FUNCTION SAMPLERS
#-------------------------------------------------------------------------------------

def get_bayes_opt_sample_lcb( gp_model, iteration, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    delta           = 0.1
    metadata        = { 'iterations': iteration, 'dimensionality': dimensionality, 'delta': delta  }
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model, metadata]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso( acquisition_lcb ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_lcb_with_der( gp_model, iteration, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    delta           = 0.1
    metadata        = { 'iterations': iteration, 'dimensionality': dimensionality, 'delta': delta  }
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model, metadata]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso( acquisition_lcb_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal


def get_bayes_opt_sample_ei( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [ gp_model ]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_ei_with_der( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [ gp_model ]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_ei_abo( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 250 )
    maxiter         = min( 500, dimensionality * 200 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei_abo ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_ei_abo_with_der( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 250 )
    maxiter         = min( 500, dimensionality * 200 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei_abo_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_el( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_el ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_el_with_der( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_el_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_el_abo( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei_abo ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_el_abo_with_der( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_ei_abo_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_mm( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_mm ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

def get_bayes_opt_sample_mm_with_der( gp_model, lower_bounds =[-1], upper_bounds=[1] ):

    # assign required variables
    dimensionality  = len( lower_bounds )
    lb              = lower_bounds
    ub              = upper_bounds
    args            = [gp_model]
    # initialize optimization procedure
    swarmsize       = min( 500, dimensionality * 100 )
    maxiter         = min( 500, dimensionality * 100 )
    minstep         = 1e-8
    minfunc         = 1e-8
    # perform optimization using PSO
    proposal, fopt  = pso(  acquisition_mm_with_derivatives ,
                            lb , ub,
                            swarmsize=swarmsize,
                            maxiter=maxiter,
                            minstep=minstep,
                            minfunc=minfunc,
                            args = args )
    return proposal

######################################################################################
#                          BAYESIAN OPTIMIZATION WRAPPERS
#-------------------------------------------------------------------------------------

def initialize_bayes_opt_model( x, y, n_epochs, first_loss ):

    # define kernel
    k_t_1       = gpflow.kernels.Matern12(input_dim=1,
                    variance=1.0, lengthscales=None,
                    active_dims=[0], ARD=False )
    k_t_2       = gpflow.kernels.RBF(     input_dim=1,
                    variance=1.0, lengthscales=None,
                    active_dims=[0], ARD=False )
    k_s         = gpflow.kernels.RBF(     input_dim=1,
                    variance=n_epochs, lengthscales=first_loss,
                    active_dims=[1], ARD=False )
    k           = k_t_1 * k_t_2 * k_s

    # define mean function - quadratic [not used]
    m1          = gpflow.mean_functions.Linear( 1.0, 0.0 )
    m2          = gpflow.mean_functions.Linear( 1.0, 0.0 )
    mean_f      = m1 * m2

    # define model
    gp_model    = gpflow.gpr.GPR( x, y, k )

    # set likelihood variance
    gp_model.likelihood.variance                = 0.0001

    # fix some hyper-parameters
    gp_model.kern.rbf_2.variance.trainable      = False
    gp_model.kern.rbf_2.lengthscales.trainable  = False

    # train model
    gp_model.optimize()

    return gp_model

def initialize_bayes_opt_model_with_der( x_input, y_loss, y_grads, 
                                                        n_epochs, first_loss  ): 

    # define kernel
    k_t_1       = gpflow.kernels.Matern12(input_dim=1,
                    variance=1.0, lengthscales=None,
                    active_dims=[0], ARD=False )
    k_t_2       = gpflow.kernels.RBF(     input_dim=1,
                    variance=1.0, lengthscales=None,
                    active_dims=[0], ARD=False )
    k_s         = gpflow.kernels.RBF(     input_dim=1,
                    variance=n_epochs, lengthscales=first_loss,
                    active_dims=[1], ARD=False )
    k_1         = k_t_1 * k_t_2 * k_s
    coreg       = gpflow.kernels.Coregion(1, output_dim=2, 
                    rank=1, active_dims=[2] )
    k           = k_1 * coreg

    # prepare coregion data, assume: x (Nx2), y_loss and y_grads (Nx1)
    x_1         = np.hstack( ( x_input, np.zeros_like(y_loss) ) )
    x_2         = np.hstack( ( x_input,  np.ones_like(y_loss) ) )
    x           = np.vstack( ( x_1  ,  x_2  )  )

    y_1         = np.hstack( ( y_loss,  np.zeros_like(y_loss) ) )
    y_2         = np.hstack( ( y_grads,  np.ones_like(y_loss) ) )
    y           = np.vstack( ( y_1  ,  y_2  )  )

    # define model
    gp_model    = gpflow.gpr.GPR( x, y, k )

    # randomly initlaize W to avoid instability
    gp_model.kern.coregion.W                    = np.random.randn(2, 1)

    # set likelihood variance
    gp_model.likelihood.variance                = 0.0001

    # fix some hyper-parameters
    gp_model.kern.rbf_2.variance.trainable      = False
    gp_model.kern.rbf_2.lengthscales.trainable  = False

    # train model
    gp_model.optimize()

#----------

def get_bayes_opt_proposal_ei( gp_model, lb, ub   ):

    sample     =  get_bayes_opt_sample_ei( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample


def get_bayes_opt_proposal_ei_with_der( gp_model, lb, ub ):

    sample     =  get_bayes_opt_sample_ei_with_der( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample


def get_bayes_opt_proposal_ei_abo( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_ei_abo( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_ei_abo_with_der( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_ei_abo_with_der( gp_model, 
                                        lower_bounds = lb, upper_bounds = ub )
    return sample

#----------

def get_bayes_opt_proposal_el( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_el( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_el_with_der( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_el_with_der( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_el_abo( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_el_abo( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_el_abo_with_der( gp_model, lb, ub   ):

    sample      = get_bayes_opt_sample_el_abo_with_der( gp_model, lower_bounds = lb,
                                                            upper_bounds = ub )
    return sample

#----------

def get_bayes_opt_proposal_lcb( gp_model, iteration, lb, ub  ):

    sample      = get_bayes_opt_sample_lcb( gp_model, iteration,
                                        lower_bounds =lb, upper_bounds=ub )
    return sample

def get_bayes_opt_proposal_lcb_with_der( gp_model, iteration, lb, ub  ):

    sample      = get_bayes_opt_sample_lcb_with_der( gp_model, iteration,
                                        lower_bounds =lb, upper_bounds=ub )
    return sample

def get_bayes_opt_proposal_lcb_abo( gp_model, iteration, lb, ub  ):

    sample      = get_bayes_opt_sample_lcb( gp_model, iteration,
                                        lower_bounds = lb, upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_lcb_abo_with_der( gp_model, iteration, lb, ub  ):

    sample      = get_bayes_opt_sample_lcb_with_der( gp_model, iteration,
                                        lower_bounds = lb, upper_bounds = ub )
    return sample

#----------

def get_bayes_opt_proposal_mm( gp_model, lb, ub ):

    sample      = get_bayes_opt_sample_mm( gp_model, lower_bounds = lb,
                                                    upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_mm_with_der( gp_model, lb, ub ):

    sample      = get_bayes_opt_sample_mm_with_der( gp_model, lower_bounds = lb,
                                                    upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_mm_abo( gp_model, lb, ub  ):

    sample      = get_bayes_opt_sample_mm( gp_model, lower_bounds = lb,
                                            upper_bounds = ub )
    return sample

def get_bayes_opt_proposal_mm_abo_with_der( gp_model, lb, ub  ):

    sample      = get_bayes_opt_sample_mm_with_der( gp_model, lower_bounds = lb,
                                            upper_bounds = ub )
    return sample

#----------

def update_bayes_opt_model( gp_model, x , y ):

    # update x
    temp_x      = np.vstack( (gp_model.X.value , x) )
    gp_model.X  = temp_x

    # update y
    temp_y      = np.vstack( (gp_model.Y.value , y) )
    gp_model.Y  = temp_y

    # re-train model
    gp_model.optimize()

    return gp_model


def update_bayes_opt_model_with_der( gp_model, x_input , y_loss, y_grads ):

    # update x
    x_1         = np.hstack( ( x_input, np.zeros_like(y_loss) ) )
    x_2         = np.hstack( ( x_input,  np.ones_like(y_loss) ) )
    x           = np.vstack( ( x_1  ,  x_2  )  )

    temp_x      = np.vstack( (gp_model.X.value , x) )
    gp_model.X  = temp_x

    # update y
    y_1         = np.hstack( ( y_loss,  np.zeros_like(y_loss) ) )
    y_2         = np.hstack( ( y_grads,  np.ones_like(y_loss) ) )

    y           = np.vstack( ( y_1  ,  y_2  )  )
    temp_y      = np.vstack( (gp_model.Y.value , y) )
    gp_model.Y  = temp_y

    # re-train model
    gp_model.optimize()

    return gp_model


######################################################################################
#                            AUXILLARY TEST FUNCTIONS
#-------------------------------------------------------------------------------------
def plot(m):
    xx          = np.linspace(-0.1, 1.1, 100)[:,None]
    mean, var   = m.predict_y(xx)

    plt.figure( figsize=(6, 3) ) #
    plt.plot(   x  , y, 'kx',   mew=2 )
    plt.plot(   xx , mean, 'b', lw=2  )
    plt.fill_between( xx[:,0], mean[:,0] - 2*np.sqrt( var[:,0] ),
                               mean[:,0] + 2*np.sqrt( var[:,0] ),
                               color='blue', alpha=0.2       )
    plt.xlim(-0.1, 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['data', 'posterior mean', 'posterior varaince'])
    plt.show(block=True)


def plot2d(m):
    xx1         = np.linspace(0, 100, 10)[:,None]
    xx2         = np.linspace(0, 3.0, 10)[:,None]
    x1, x2      = np.meshgrid( xx1 , xx2 )
    x1_         = x1.reshape((100,1))
    x2_         = x2.reshape((100,1))
    xx          = np.hstack( (x1_, x2_) )
    mean, var   = m.predict_y(xx)
    z1          = mean
    z2          = var
    z           = z1.reshape( x1.shape )

    # Plot the surface.
    fig         = plt.figure()
    ax          = fig.add_subplot(111, projection='3d')
    surf        = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.savefig("0.png")
    # plt.show(block=True)

    return x1, x2, z


######################################################################################
#                        DEFAULT EXECUTION /  TESTS
#-------------------------------------------------------------------------------------

# use dummy data

N         = 200
x1        = np.random.rand( N , 1 )
x2        = np.linspace(0, 10, N)[:,None]
x1.sort(    axis=0)

y         = np.sin(12*x1) + 0.66*np.cos(25*x1) + np.random.randn(N,1)*0.01 + 0

y_         = np.sin(12*x1) + 0.66*np.cos(25*x1) + np.sin(12*x2) + \
                      0.66*np.cos(25*x2) + np.random.randn(N,1)*0.01 + 0

# plt.figure( )
# plt.plot(   x1, y, 'k', mew=3, lw=2 )
# plt.xlabel( 'x')
# plt.ylabel( 'y')
# plt.legend( 'True')
# plt.title(  'True function')
# plt.savefig("0.png")

# plt.show(   block=True)

x          = np.hstack( (x2, x1) )

first_loss = np.min( y_ )
n_epochs   = 10
# model_1    = initialize_bayes_opt_model(x, y, n_epochs, first_loss)

#-------------------------------------------------------------------------------------

# use real epoch data

data       = sio.loadmat(
                  '../logs/final0/softmax-nesterov-mnist-16-March-2018-02-01-03-AM-with-gradients-25-epochs')
loss       = data['loss_epochs']
grads      = data['gradients_epochs']
eta        = data['eta_epochs']
iters      = np.arange(25) + 1

loss       = loss.reshape(  (25,1) )
grads      = grads.reshape( (25,1) )
eta        = eta.reshape(   (25,1) )
iters      = iters.reshape( (25,1) )

x          = np.hstack( (iters, eta) )
y          = loss
y_loss     = loss
y_grads    = grads

n_epochs   = 25
first_loss = loss[0]
gp_model   = initialize_bayes_opt_model( x, y, n_epochs, first_loss )

# x, y, z    = plot2d(gp_model)

lb         = [24.99, 0.0001]
ub         = [25.00, 0.001]

sample1    = get_bayes_opt_proposal_ei(  gp_model, lb, ub   )
sample2    = get_bayes_opt_proposal_el(  gp_model, lb, ub   )
sample3    = get_bayes_opt_proposal_lcb( gp_model, 25, lb, ub   )
sample4    = get_bayes_opt_proposal_mm(  gp_model, lb, ub   )

sample     = sample1
print sample
sample     = sample[:,None].T # add extra axis
m_, v_     = gp_model.predict_y(sample)
print m_
updated    = update_bayes_opt_model(gp_model, sample, m_)

#-------------------------------------------------------------------------------------

# Test derivative versions 
print( '\nDEALING WITH DERIVATIVES' )

gp_model   = initialize_bayes_opt_model_with_der( x, y_loss, y_grads, 
                                                          n_epochs, first_loss )

# x, y, z    = plot2d(gp_model)

# lb         = [ 24.99, 0.0001 ]
# ub         = [ 25.00, 0.0010 ]

# sample1    = get_bayes_opt_proposal_ei_with_der(  gp_model, lb, ub   )
# sample2    = get_bayes_opt_proposal_el_with_der(  gp_model, lb, ub   )
# sample3    = get_bayes_opt_proposal_lcb_with_der( gp_model, 25, lb, ub   )
# sample4    = get_bayes_opt_proposal_mm_with_der(  gp_model, lb, ub   )

# sample     = sample1
# print sample
# sample     = sample[:,None].T # add extra axis
# m_, v_     = gp_model.predict_y(sample)
# print m_
# updated    = update_bayes_opt_model_with_der(gp_model, sample, m_, m_)



#-------------------------------------------------------------------------------------
# use methods to solve standard Bayesianoptimization problem



