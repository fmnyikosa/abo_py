# -*- coding: utf-8 -*-
# This module contains the acquisition functions for Bayesian optimisation (BO)
# and adaptive Bayesian Optimisation (ABO). It contains the following
# acquisition functions:
#
#   Expected Improvement (EI) [versions 1 and 2]
#   Expected Improvement (EI) ABO
#   Expected Loss (EL)
#   Expected Loss (EL) ABO
#   Upper Confidence Bounds (UCB)
#   Upper Confidence Bounds (UCB)
#   Lower Confidence Bounds (LCB)
#   Lower Confidence Bounds (LCB) ABO
#   Maximum Mean (MM)
#   Maximum Mean (MM) ABO
#   Probability of Improvement (PI)
#   Probability of Improvement (PI) ABO
#
# Copyright (c) by Favour Mandanji Nyikosa <favour@nyikosa.com>, 2017-MAY-29

import numpy as numpy
import scipy.linalg as linalg
import sys

sys.path.append("..")

def acquisitionEI1(x0, metadata):
    """
    Expected Improvement acquistion function for Bayesian optimization
    from Brochu et. al (2010) [A Tutorial on Bayesian Optimzation of Expensive
    Cost Function, with Application to Active User Modelling and Hierachical
    Reinforcement Learning]. It is based on the maximization formalisation
    of the global optimisation problem,

                              MAX f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

      el, g, post_metadata = acquisitionEI1(x0, metadata)

      where

          x0:             Sample position [N X DIM] vector or matrix
          metadata:       Struct of metadata from a GP training
          ei:             Expected Improvement (NB: The higher, the better)
          g:              Gradient of Expected Improvement function
          post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt            = metadata.xt
    yt            = metadata.yt
    hyp           = metadata.training_hyp
    gpModel       = metadata.gpDef
    mean_,var_    = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)
    if metadata.abo == 1:
        threshold = yt(end)
    else:
        threshold = max(yt)
    sDev          = sqrt(var_)
    if sDev > 0:
        Z         = ( mean_ - threshold ) / sDev
        CDF       = normcdf(Z)
        PDF       = normpdf(Z)
        ei        = ( (mean_- threshold ) * CDF ) + (sDev * PDF)
        ei        = -ei
    else:
         ei       = 0
    if nargout == 2:
        #  gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        mean_h, var_h   = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        Z_h             = ( mean_h - threshold ) / sDev_h
        CDF_h           = normcdf(Z_h)
        PDF_h           = normpdf(Z_h)
        ei_h            = ( (mean_h - threshold ) * CDF_h ) + (sDev_h * PDF_h)
        ei_h            = -ei_h
        g               = (ei_h - ei) / h
    if nargout > 2:
        g               = 0
    return ei, g, post_metadata

def acquisitionEI1_ABO(x0, metadata):
    """
    Expected Improvement acquistion function for adaptive Bayesian optimization
    from Brochu et. al (2010) [A Tutorial on Bayesian Optimzation of Expensive
    Cost Function, with Application to Active User Modelling and Hierachical
    Reinforcement Learning]. It is based on the maximization formalisation
    of the global optimisation problem,

                             MAX f(t, x), where t known apriori.

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

     [el, g, post_metadata] = acquisitionEI1_ABO(x0, metadata)

     where

         x0:             Sample position [N X DIM] vector or matrix
         metadata:       Struct of metadata from a GP training
         ei:             Expected Improvement (NB: The higher, the better)
         g:              Gradient of Expected Improvement function
         post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time           = metadata.current_time_abo
    x0                     = [current_time, x0]
    [ei, g, post_metadata] = acquisitionEI1(x0, metadata)
    return ei, g, post_metadata

def acquisitionEI2(x0, metadata):
    """
    Expected Improvement acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MIN f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    ei, g, post_metadata = acquisitionEI2(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ei:             Expected impprovement (NB: The higher, the better)
        g:              Gradient of EI function
        post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt           = metadata.xt
    yt           = metadata.yt
    hyp          = metadata.training_hyp
    gpModel      = metadata.gpDef
    mean_, var_  = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)
    if metadata.abo == 1:
        threshold= yt(end)
    else:
        threshold= min(yt)
    sDev         = sqrt(var_)
    if sDev > 0:
        Z        = ( threshold - mean_  ) / sDev
        CDF      = normcdf(Z)
        PDF      = normpdf(Z)
        ei       = sDev *  (  (Z * CDF) + PDF )
        ei       = -ei
    else:
        ei       = 0
    if nargout == 2:
        #  gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        [mean_h, var_h] = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        Z_h             = ( threshold - mean_h  ) / sDev_h
        CDF_h           = normcdf(Z_h)
        PDF_h           = normpdf(Z_h)
        ei_h            = sDev_h *  (  (Z_h * CDF_h) + PDF_h )
        ei_h            = -ei_h
        g               = (ei_h - ei) / h
    if nargout > 2:
        g               = 0
    return ei, g, post_metadata

def acquisitionEI2_ABO(x0, metadata):
    """
    Expected Improvement acquistion function for adaptive Bayesian optimization
    from Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine
    Learning Algorithms]. It is based on the maximization formalisation of the
    global optimisation problem,

                          MIN f(t, x), where t known apriori.

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

      [ei, g, post_metadata]l = acquisitionEI2_ABO(x0, metadata)

      where

          x0:             Sample position [N X DIM] vector or matrix
          metadata:       Struct of metadata from a GP training
          ei:             Expected impprovement (NB: The higher, the better)
          g:              Gradient of EI function
          post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time           = metadata.current_time_abo
    x0                     = [current_time, x0]
    [ei, g, post_metadata] = acquisitionEI2(x0, metadata)
    return ei, g, post_metadata

def acquisitionEL(x0, metadata):
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

    [el, g, post_metadata]l = acquisitionEL(x0, metadata)

    where

       x0:             Sample position [N X DIM] vector or matrix
       metadata:       Struct of metadata from a GP training
       el:             Expected loss (NB: The lower, the better)
       g:              Gradient of the expected loss function
       post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt            = metadata.xt
    yt            = metadata.yt
    hyp           = metadata.training_hyp
    gpModel       = metadata.gpDef
    mean_,var_    = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)
    if metadata.abo == 1:
        threshold = yt(end)
    else:
        threshold = min(yt)
    sDev          = sqrt(var_)
    CDF           = normcdf(threshold, mean_, sDev )
    PDF           = normpdf(threshold, mean_, sDev )
    el            = threshold + ( (mean_ - threshold) * CDF ) - ( sDev * PDF)
    if nargout == 2:
        # gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        [mean_h, var_h] = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        CDF_h           = normcdf(threshold, mean_h, sDev_h )
        PDF_h           = normpdf(threshold, mean_h, sDev_h )
        el_h            = threshold + ( (mean_h - threshold) * CDF_h ) \
                                                       - ( sDev_h * PDF_h)
        g               = (el_h - el) / h
    if nargout > 2:
        g               = 0
    return el, g, post_metadata

    def acquisitionEL_ABO(x0, metadata):
    """
    Expected Loss acquistion function for adaptive Bayesian optimization from
    Osborne et. al (2011) [Gaussian Processes for Global Optimization].
    It is based on the minimization formalisation of the global optimisation
    problem,

                     MIN f(t, x), where t known apriori.

    NOTE: To find optimal position to sample, MINIMIZE this acquisition
    function. This is unlike the expected improvement which you would need to
    maximize to obtain the optimal sample position.

    Usage:

    [el, g, post_metadata]l = acquisitionEL_ABO(x0, metadata)

    where

      x0:             Sample position [N X DIM] vector or matrix
      metadata:       Struct of metadata from a GP training
      el:             Expected loss (NB: The lower, the better)
      g:              Gradient of the expected loss function
      post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time           = metadata.current_time_abo
    x0                     = [current_time, x0]
    el, g, post_metadata   = acquisitionEL(x0, metadata)
    return el, g, post_metadata

    def acquisitionUCB(x0, metadata):
    """
    Upper Confidence Bound acquistion function for Bayesian optimization from
    Srinivas et. al (2010) [Gaussian Processes for Global Optimization].
    It is based on the maximization formalisation of the global optimisation
    problem,

                       MAX f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    [ucb, g, post_metadata] = acquisitionUCB(x0, metadata)

    where

    x0:             Sample position [N X DIM] vector or matrix
    metadata:       Struct of metadata from a GP training
    ucb:            Upper confidence bounds (NB: The higher, the better)
    g:              Gradient of UCB function
    post_metadata:  Post-processing struct metadata from GP prediction
    """
    g                = 0
    xt               = metadata.xt
    yt               = metadata.yt
    hyp              = metadata.training_hyp
    gpModel          = metadata.gpDef
    mean_,var_       = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)
    iterations       = metadata.iterations
    dimensionality   = metadata.dimensionality
    delta            = metadata.delta
    if iterations > 1:
        kappa        = calculateUCBKappa1(iterations, dimensionality, delta)
    else:
        kappa        = 1
    practical_factor = (1/5)  # from Srinivas et al. (2010)
    kappa            = kappa .* practical_factor
    sDev             = sqrt(var_)
    ucb              = mean_ + (kappa * sDev)
    ucb              = -ucb
    post_metadata    = ps
    if nargout == 2:
        # gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        [mean_h, var_h] = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        ucb_h           = mean_h + (kappa * sDev_h)
        ucb_h           = -ucb_h
        g               = (ucb_h - ucb)  / h
    return ucb, g, post_metadata

def acquisitionUCB_ABO(x0, metadata):
    """
    Upper Confidence Bound acquistion function for adaptive Bayesian
    optimization from Srinivas et. al (2010) [Gaussian Processes for Global
    Optimization]. It is based on the maximization formalisation of the global
    optimisation problem,

                         MAX f(t, x), where t known apriori.

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition
       function.

    Usage:

    [ucb, g, post_metadata] = acquisitionUCB_ABO(x0, metadata)

    where

       x0:             Sample position [N X DIM] vector or matrix
       metadata:       Struct of metadata from a GP training
       ucb:            Upper confidence bounds (NB: The higher, the better)
       g:              Gradient of UCB function
       post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time            = metadata.current_time_abo
    x0                      = [current_time, x0]
    [ucb, g, post_metadata] = acquisitionUCB(x0, metadata)
    return ucb, g, post_metadata

def acquisitionLCB(x0, metadata):
    """
    Lower Confidence Bound acquistion function for Bayesian optimization from
    Srinivas et. al (2010) [Gaussian Processes for Global Optimization].
    It is based on the maximization formalisation of the global optimisation
    problem,

                               MIN f(x).

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function

    Usage:

    [ucb, g, post_metadata] = acquisitionLCB(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ucb:            Upper confidence bounds (NB: The higher, the better)
        g:              Gradient of UCB function
        post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt               = metadata.xt
    yt               = metadata.yt
    hyp              = metadata.training_hyp
    gpModel          = metadata.gpDef

    [mean_,var_,~,~,ps] = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)

    iterations       = metadata.iterations
    dimensionality   = metadata.dimensionality
    delta            = metadata.delta
    kappa            = calculateUCBKappa1(iterations, dimensionality, delta)
    practical_factor = (1/5)  #  from Srinivas et al. (2010)
    kappa            = kappa * practical_factor

    kappa            = 1

    sDev             = sqrt(var_)
    lcb              = mean_ - (kappa * sDev)
    lcb              = lcb

    post_metadata    = ps

    if nargout == 2:
        #  gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        [mean_h, var_h] = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        lcb_h           = mean_h - (kappa * sDev_h)
        lcb_h           = lcb_h
        g               = (lcb_h - lcb) /h
    if nargout > 2:
        g               = 0
    return lcb, g, post_metadata

def acquisitionLCB_ABO(x0, metadata):
    """
    Lower Confidence Bound acquistion function for adaptive Bayesian
    optimization from Srinivas et. al (2010) [Gaussian Processes for Global
    Optimization]. It is based on the maximization formalisation of the global
    optimisation problem,

                     MIN f(t, x), where t is known apriori.

    NOTE: To find optimal position to sample, MINIMIZE this acquisition function

    Usage:

    [ucb, g, post_metadata] = acquisitionLCB_ABO(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ucb:            Upper confidence bounds (NB: The higher, the better)
        g:              Gradient of UCB function
        post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time            = metadata.current_time_abo
    x0                      = [current_time, x0]
    [lcb, g, post_metadata] = acquisitionLCB(x0, metadata)
    return lcb, g, post_metadata

def acquisitionMM(x0, metadata):
    """
    Maximum Mean acquistion function for Bayesian optimization from
    Snoek et. al (2012) [Practiacal Bayesian Optimzation of MAchine Learning
    Algorithms]. It is based on the maximization formalisation of the global
    optimisation problem,

                            MAX f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    [mean_, g, post_metadata] = acquisitionMM(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ei:             Expected impprovement (NB: The higher, the better)
        g:              Gradient of EI function
        post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt           = metadata.xt
    yt           = metadata.yt
    hyp          = metadata.training_hyp
    gpModel      = metadata.gpDef
    mean_        = getGPResponse(x0,xt,yt,hyp,gpModel,metadata)
    if nargout == 2:
        #  gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        mean_h          = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        g               = (mean_h - mean_)./h
    if nargout > 2:
        g               = 0
    return mean_, g, post_metadata

def acquisitionMM_ABO(x0, metadata):
    """
    Maximum Mean acquistion function for adaptive Bayesian optimization
    from Brochu et. al (2010) [A Tutorial on Bayesian Optimzation of Expensive
    Cost Function, with Application to Active User Modelling and Hierachical
    Reinforcement Learning]. It is based on the maximization formalisation
    of the global optimisation problem,

                            MAX f(t, x), where t can known apriori.

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    [mm, g, post_metadata] = acquisitionMM_ABO(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ei:             Expected Improvement (NB: The higher, the better)
        g:              Gradient of Expected Improvement function
        post_metadata:  Post-processing struct metadata from GP prediction

    """
    current_time           = metadata.current_time_abo
    x0                     = [current_time, x0]
    [mm, g, post_metadata] = acquisitionMM(x0, metadata)
    return mm, g, post_metadata

def acquisitionPI(x0, metadata):
    """
    Probability of  Improvement acquistion function for Bayesian optimization
    from Snoek et. al (2012) [Practiacal Bayesian Optimzation of Machine
    Learning Algorithms]. It is based on the maximization formalisation of the
    global optimisation problem,

                            MAX f(x).

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    [pi_, g, post_metadata] = acquisitionPI(x0, metadata)

    where

        x0:             Sample position [N X DIM] vector or matrix
        metadata:       Struct of metadata from a GP training
        ei:             Expected impprovement (NB: The higher, the better)
        g:              Gradient of EI function
        post_metadata:  Post-processing struct metadata from GP prediction
    """
    xt           = metadata.xt
    yt           = metadata.yt
    hyp          = metadata.training_hyp
    gpModel      = metadata.gpDef

    [mean_,var_,~,~,post_metadata]=getGPResponse(x0,xt,yt,hyp,gpModel,metadata)

    if metadata.abo == 1:
        threshold=yt(end)
    else:
        threshold=max(yt)

    sDev         = sqrt(var_)

    Z            = ( mean_  - 2 * threshold ) / sDev
    pi_          = -normcdf(Z)

    if nargout == 2:
        #  gradient by finite differences (difference h)
        h               = eps
        x0_h            = x0 + h
        [mean_h, var_h] = getGPResponse(x0_h,xt,yt,hyp,gpModel,metadata)
        sDev_h          = sqrt(var_h)
        Z_h             = ( mean_h  - threshold ) / sDev_h
        pi_h           = -normcdf(Z_h)
        g               = (pi_h - pi_) / h

    if nargout > 2:
        g               = 0

    return pi_, g, post_metadata


def acquisitionPI_ABO(x0, metadata):
    """
    Probability of Improvement acquistion function for adaptive Bayesian
    from Brochu et. al (2010) [A Tutorial on Bayesian Optimzation of Expensive
    Cost Function, with Application to Active User Modelling and Hierachical
    Reinforcement Learning]. It is based on the maximization formalisation
    of the global optimisation problem,

                           MAX f(t, x), where t can be known apriori.

    NOTE: To find optimal position to sample, MAXIMIZE this acquisition function

    Usage:

    [pi, g, post_metadata] = acquisitionPI_ABO(x0, metadata)

    where

       x0:             Sample position [N X DIM] vector or matrix
       metadata:       Struct of metadata from a GP training
       ei:             Expected Improvement (NB: The higher, the better)
       g:              Gradient of Expected Improvement function
       post_metadata:  Post-processing struct metadata from GP prediction
    """
    current_time           = metadata.current_time_abo
    x0                     = [current_time, x0]
    [pi, g, post_metadata] = acquisitionPI(x0, metadata)
    return pi, g, post_metadata
