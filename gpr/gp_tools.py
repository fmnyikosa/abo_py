# -*- coding: utf-8 -*-
#
# This module contains the wrapper functions and tools for Gaussian process (GP)
# regression using GPflow and some custom code.
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com> 29-MAY-2017

# Gaussian process regression (GPR) wrapper
def predictGPR(x_star, gpModel):
    """
    This function trains a Gaussian process model and calculates the posterior
    mean and variance for Gaussian process regression (GPR).
    Usage:
        mean_, var_, post_meta = predictGPR(x_star, gpModel)
    where:
            x_star:     query input; either <1xDIM> or <NXDIM>
            gpModel:    GPflow model object or dict
            settings:   settings dictionary
            mean_:      posterior mean ; either <1xDIM> or <NXDIM>
            var_:       posterior variance; either <1xDIM> or <NXDIM>
            post_meta:  post processing metadata dictinary
    """
    # Train model by minimizing negative log marginal likelihood
    gpModel.optimize()
    # Predict
    mean_, var_ = gpModel.predict_y(x_star)
    return        mean_, var_

# Gaussian process regression (GPR) wrapper with no traning
def getGPResponse(x_star, gpModel):
    """
    This function calculates the posterior mean and variance for Gaussian
    process regression (GPR) without training. Usage:
        mean_, var_, post_meta = predictGPR(x_star, gpModel)
    where:
            x_star:     query input; either <1xDIM> or <NXDIM>
            gpModel:    GPflow model object or dict
            settings:   settings dict
            mean_:      posterior mean ; either <1xDIM> or <NXDIM>
            var_:       posterior variance; either <1xDIM> or <NXDIM>
            post_meta:  post processing metadata dictinary
    """
    mean_, var_ = gpModel.predict_y(x_star)
    return        mean_, var_

# Plot 1 dimensional GPR
def plotGPR1D(x_star, mean_, var_, x, y, figsize_=(12, 6), xlim_=(0,1.1),\
                         xlabel_ = 'x', ylabel_ = 'y', title_ = 'GP Regression'):
    """
    This function plots the posterior mean and variance for Gaussian
    process regression (GPR). Usage:
        plotGPR1D(x_star, mean_, var_, X, Y, figsize=(12, 6))
    where:
            x_star:     query input; either <1xDIM> or <NXDIM>
            mean_:      posterior mean ; either <1xDIM> or <NXDIM>
            var_:       posterior variance; either <1xDIM> or <NXDIM>
            x:          training input data
            y:          training targets
            [figure options]: figsize_, xlim_, ylim_, xlabel_, ylabel_, title_
    """
    pyplot.figure(      figsize=figsize_)
    pyplot.plot(x,   y, 'kx',      mew=2)
    pyplot.plot(x_star,mean_, 'b',  lw=2)

    lower_b = mean_[:,0] - 2*np.sqrt(var_[:,0])
    upper_b = mean_[:,0] + 2*np.sqrt(var_[:,0])

    pyplot.fill_between(x_star[:,0],lower_b,upper_b,color='blue',alpha=0.2)
    pyplot.xlim(xlim_[0], xlim_[1])
    pyplot.xlabel(xlabel_)
    pyplot.ylabel(ylabel_)
    pyplot.title(title_)
    pyplot.show()
