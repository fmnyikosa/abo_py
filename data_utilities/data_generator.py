# -*- coding: utf-8 -*-
# This module contains methods that generate initial data to be used by the test
# objective functions for Bayesian optimization or other global optimisation
# methods. It contains the following methods:
#
#   boundRandomData:             fix data to bounds
#
#   getInitialInputFunctionData: get data from Latin hypercube
#   getInitialAckleyData:        get initial data from Branin function
#   getInitialBraninData:        get initial data from Branin function
#   getInitialBraninModData:     get initial modified Branin data
#   getInitialBraninSCData:      get initial scaled Branin data
#   getInitialCamel6Data:        get initial 6-hump Camelback data
#   getInitialEggholderData:     get initial Eggholder data
#   getInitialHartmann3Data:     get initial Hartmann3 data
#   getInitialHartmann6Data:     get initial Hartmann6 data
#   getInitialHartmann6SCData:   get initial scaled Hartmann6 data
#   getInitialStybTangData:      get initial Styblinski-Tang data
#
#   getInitialColvilleData:      get initial Colville data
#   getInitialGoldpriceData:     get initial Goldstein-Price data
#   getInitialGoldpriceSCData:   get initial scaled Goldstein-Price data
#   getInitialGriewankData:      get initial Griewank data
#   getInitialHartmann4Data:     get initial Hartmann4 data
#   getInitialRastriginData:     get initial Rastrigin data
#   getInitialShekelData:        get initial Shekel data
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com>, 27-MAY-2017

import pyDOE                      as DOE
import numpy                      as numpy
import scipy                      as scipy
from . import objective_functions as ObjectiveFunctions

def boundRandomData(random_data, lower_bound, upper_bound):
    """
    This function fixes data to set bounds.
    """
    bounded_data = lower_bound + random_data * (upper_bound - lower_bound)
    return       bounded_data

def getInitialInputData(num_points, dim, lower_b, upper_b):
    """
    This function gets input test objective function data from a Latin hypercube
    design. Usage:

    initial_data = getInitialInputFunctionData(num_points,dim,lower_b,upper_b)

            num_points:     number of datapoints neeeded (1 x 1)
            dim:            dimensionality (1 x 1)
            lower_b:        lower bound (1 x 1) or (num_points x 1)
            upper_b:        upper bound (1 x 1) or (num_points x 1)
            initial_data:   datapoints generateted (number_of_points x dim)
    """
    # get data from Latin hypercube
    random_data  = DOE.doe_lhs.lhs(dim, samples=num_points)
    # bound it
    initial_data = boundRandomData(random_data, lower_b, upper_b)
    return       initial_data

def getInitialAckleyData(num_points, dim):
    """
    Gets the initial randomly generated Nd data for Ackley function.
    Usage:

           X, y = getInitialAckleyData(num_points, dim)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           [X, y]:     datapoints generated (number_of_points * 2)

    Info: Ackley function is usually evaluated on the
          square xi \in [-10, 10]
    """
    lower_b   = -10
    upper_b   =  10

    X         = getInitialInputData(num_points, dim, lower_b, upper_b)
    y         = ObjectiveFunctions.ackley_bulk(           X)
    return    X, y

def getInitialAckleyDataABO( num_points, dim ):
    """
    Gets the initial randomly generated Nd data for Ackley function.
    Usage:

           X, y = getInitialAckleyDataABO(num_points, dim, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           max_time:   maximum time tag for intial data
           [X, y]:     datapoints generated (number_of_points * 2)

    Info: Ackley function is usually evaluated on the
          square xi \in [-10, 10]
    """
    lower_b   = -10
    upper_b   =  10

    dim_      = dim - 1
    x1        = getInitialInputData(num_points, 1, lower_b,   max_time)
    x2        = getInitialInputData(num_points, dim_, lower_b, upper_b)
    X         = numpy.concatenate((x1, x2),                     axis=1)
    y         = ObjectiveFunctions.ackley_bulk(             X)
    return    X, y


def getInitialBraninData(num_points):
    """
    Gets the initial randomly generated 2d data for Branin function.
    Usage:

    X, y = getInitialBraninData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)

    Info: Branin function is usually evaluated on the
          square x1 \in [-5, 10], x2 \in [0, 15]
    """
    lower_b1  = -5
    upper_b1  = 10
    lower_b2  = 0
    upper_b2  = 15

    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.branin_bulk(X)
    return    X, y

def getInitialBraninDataABO(num_points, max_time):
    """
    Gets the initial randomly generated 2d data for Branin function.
    Usage:

    X, y = getInitialBraninData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)

    Info: Branin function is usually evaluated on the
          square x1 \in [-5, 10], x2 \in [0, 15]
    """
    lower_b1  = -5
    upper_b1  = max_time
    lower_b2  = 0
    upper_b2  = 15

    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.branin_bulk(X)
    return X, y

def getInitialBraninModData(num_points):
    """
    Gets the initial randomly generated 2d data for modified Branin function.
    From Forrester, A., Sobester, A., & Keane, A. (2008).
    Engineering design via surrogate modelling: a practical guide. Wiley.

    Usage:

    X, y = getInitialBraninModData(num_points)

         num_points: number of datapoints neeeded (1 x 1)
         [X, y]:     datapoints generated (number_of_points * 2)

    Info: For the purpose of Kriging prediction, Forrester et al. (2008)
    use a modified form of the Branin-Hoo function, in which they add a term
    5x1 to the response. As a result, there are two local minima and only one
    global minimum, making it more representative of engineering functions.
    """
    lower_b1  = -5
    upper_b1  = 10
    lower_b2  = 0
    upper_b2  = 15
    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.branin_modified_bulk(X)
    return X, y


def getInitialBraninModDataABO(num_points, max_time):
    """
    Gets the initial randomly generated 2d data for modified Branin function.
    From Forrester, A., Sobester, A., & Keane, A. (2008).
    Engineering design via surrogate modelling: a practical guide. Wiley.

    Usage:

    X, y = getInitialBraninModData(num_points)

         num_points: number of datapoints neeeded (1 x 1)
         max_time:   maximum time for initial data
         [X, y]:     datapoints generated (number_of_points * 2)

    Info: For the purpose of Kriging prediction, Forrester et al. (2008)
    use a modified form of the Branin-Hoo function, in which they add a term
    5x1 to the response. As a result, there are two local minima and only one
    global minimum, making it more representative of engineering functions.
    """
    lower_b1  = -5
    upper_b1  = max_time
    lower_b2  = 0
    upper_b2  = 15
    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.branin_modified_bulk(X)
    return X, y

def getInitialBraninModDataABO(num_points, max_time):
    """
    Gets the initial randomly generated 2d data for modified Branin function.
    From Forrester, A., Sobester, A., & Keane, A. (2008).
    Engineering design via surrogate modelling: a practical guide. Wiley.

    Usage:

    X, y = getInitialBraninModData(num_points)

         num_points: number of datapoints neeeded (1 x 1)
         max_time:   maximum time for initial data
         [X, y]:     datapoints generated (number_of_points * 2)

    Info: For the purpose of Kriging prediction, Forrester et al. (2008)
    use a modified form of the Branin-Hoo function, in which they add a term
    5x1 to the response. As a result, there are two local minima and only one
    global minimum, making it more representative of engineering functions.
    """
    lower_b1  = -5
    upper_b1  = max_time
    lower_b2  = 0
    upper_b2  = 15
    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.branin_modified_bulk(X)
    return X, y

def getInitialBraninSCData(num_points):
    """
    Gets the initial randomly generated 2d data for rescaled Branin function.
    From Picheny, V., Wagner, T., & Ginsbourger, D. (2012).
    A benchmark of kriging-based infill criteria for noisy optimization.

    [X, y] = getInitialBraninSCData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)

       Info: Rescaled Branin function is evaluated on the square x1,x2 \in [0,1]
    """
    lower_b   = 0
    upper_b   = 15
    X         = getInitialInputData(num_points, 2, lower_b, upper_b)
    y         = ObjectiveFunctions.branin_rescaled_bulk(X)
    return X, y

def getInitialBraninSCDataABO(num_points, max_time):
    """
    Gets the initial randomly generated 2d data for rescaled Branin function.
    From Picheny, V., Wagner, T., & Ginsbourger, D. (2012).
    A benchmark of kriging-based infill criteria for noisy optimization.

    [X, y] = getInitialBraninSCDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)

       Info: Rescaled Branin function is evaluated on the square x1,x2 \in [0,1]
    """
    lower_x   = 0
    upper_x   = 15
    t         = getInitialInputData(num_points, 1, lower_x, max_time)
    X_        = getInitialInputData(num_points, 1, lower_x, upper_x)
    X         = numpy.concatenate( (t,X_), axis=1 )
    y         = ObjectiveFunctions.branin_rescaled_bulk(X)
    return X, y

def getInitialCamel6Data(num_points):
    """
    Gets the initial randomly generated data for 6-hump Camelback function (2d).
    Usage:

    [X, y] = getInitialCamel6Data(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_b1  = -2
    upper_b1  = 2
    lower_b2  = -1
    upper_b2  = 1
    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.camel6hump_bulk(X)
    return X, y

def getInitialCamel6DataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for 6-hump Camelback function (2d).
    Usage:

    [X, y] = getInitialCamel6DataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_b1  = -2
    upper_b1  = max_time
    lower_b2  = -1
    upper_b2  = 1
    x1        = getInitialInputData(num_points, 1, lower_b1, upper_b1)
    x2        = getInitialInputData(num_points, 1, lower_b2, upper_b2)
    X         = numpy.concatenate((x1, x2), axis=1)
    y         = ObjectiveFunctions.camel6hump_bulk(X)
    return X, y

def getInitialEggholderData(num_points):
    """
    Gets the initial randomly generated data for the Eggholder function (2d).
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -512
    upper_x  =  512
    X        = getInitialInputData(num_points, 2, lower_x, upper_x)
    y        = ObjectiveFunctions.eggholder_bulk(X)
    return X, y

def getInitialEggholderDataABO(num_points,max_time):
    """
    Gets the initial randomly generated data for the Eggholder function (2d).
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -512
    upper_x  =  512
    t        = getInitialInputData(num_points, 1, lower_x, max_time)
    X_       = getInitialInputData(num_points, 1, lower_x, upper_x)
    X        = numpy.concatenate( (t,X_), axis=1 )
    y        = ObjectiveFunctions.eggholder_bulk(X)
    return X, y

def getInitialHartmann3Data(num_points):
    """
    Gets the initial randomly generated data for the Hartmann 3d function.
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    X        = getInitialInputData(num_points, 3, lower_x, upper_x)
    y        = ObjectiveFunctions.hartmann3_bulk(X)
    return X, y

def getInitialHartmann3DataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Hartmann 3d function.
    Usage:

    [X, y] = getInitialHartmann3DataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    t        = getInitialInputData(num_points, 1, lower_x, max_time)
    X_       = getInitialInputData(num_points, 2, lower_x, upper_x)
    X        = numpy.concatenate( (t,X_), axis=1 )
    y        = ObjectiveFunctions.hartmann3_bulk(X)
    return X, y

def getInitialHartmann6Data(num_points):
    """
    Gets the initial randomly generated data for the Hartmann 6d function.
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    X        = getInitialInputData(num_points, 6, lower_x, upper_x)
    y        = ObjectiveFunctions.hartmann6_bulk(X)
    return X, y

def getInitialHartmann6DataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Hartmann 6d function.
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    t        = getInitialInputData(num_points, 1, lower_x, max_time)
    X_       = getInitialInputData(num_points, 5, lower_x, upper_x)
    X        = numpy.concatenate( (t,X_), axis=1 )
    y        = ObjectiveFunctions.hartmann6_bulk(X)
    return X, y

def getInitialHartmann6SCData(num_points):
    """
    Gets the initial randomly generated data for the scaled Hartmann 6d
    function.
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    X        = getInitialInputData(num_points, 6, lower_x, upper_x)
    y        = ObjectiveFunctions.hartmann6_rescaled_bulk(X)
    return X, y

def getInitialHartmann6SCDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the scaled Hartmann 6d
    function.
    Usage:

    [X, y] = getInitialHartmann6SCDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = 0
    upper_x  = 1
    t        = getInitialInputData(num_points, 1, lower_x, max_time)
    X_       = getInitialInputData(num_points, 5, lower_x, upper_x)
    X        = numpy.concatenate( (t,X_), axis=1 )
    y        = ObjectiveFunctions.hartmann6_rescaled_bulk(X)
    return X, y


def getInitialStybTangData(num_points):
    """
    Gets the initial randomly generated data for the Styblinski-Tang function
    (2d).
    Usage:

    [X, y] = getInitialEggholderData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -5
    upper_x  = 5
    X        = getInitialInputData(num_points, 2, lower_x, upper_x)
    y        = ObjectiveFunctions.styblinski_tang_bulk(X)
    return X, y

def getInitialStybTangDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Styblinski-Tang function
    (2d).
    Usage:

    [X, y] = getInitialStybTangDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -5
    upper_x  = 5
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 1, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    X        = X[ X[:,0].argsort() ]
    y        = ObjectiveFunctions.styblinski_tang_bulk(X)
    return X, y

#-------------------------------------------------------------------------------
#                        Added later: Oct 15th 2017
#-------------------------------------------------------------------------------


def getInitialColvilleData(num_points):
    """
    Gets the initial randomly generated data for the Colville function
    (4d).
    Usage:

    [X, y] = getInitialColvilleData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 4)
    """
    lower_x  = -10
    upper_x  = 10
    X        = getInitialInputData(num_points, 4, lower_x, upper_x)
    y        = ObjectiveFunctions.colville_bulk(X)
    return X, y

def getInitialColvilleDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Colville function
    (4d).
    Usage:

    [X, y] = getInitialColvilleDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 4)
    """
    lower_x  = -10
    upper_x  = 10
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 3, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.colville_bulk(X)
    return X, y

def getInitialGoldpriceData(num_points):
    """
    Gets the initial randomly generated data for the Goldstein-Price
    function (2d).
    Usage:

    [X, y] = getInitialGoldpriceData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -2
    upper_x  = 2
    X        = getInitialInputData(num_points, 2, lower_x, upper_x)
    y        = ObjectiveFunctions.goldprice_bulk(X)
    return X, y

def getInitialGoldpriceDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Goldstein-Price
    function (2d).
    Usage:

    [X, y] = getInitialGoldpriceDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -2
    upper_x  = 2
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 1, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.goldprice_bulk(X)
    return X, y

def getInitialGoldpriceSCData(num_points):
    """
    Gets the initial randomly generated data for the Rescaled
    Goldstein-Price function (2d).
    Usage:

    [X, y] = getInitialGoldpriceSCData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -2
    upper_x  = 2
    X        = getInitialInputData(num_points, 2, lower_x, upper_x)
    y        = ObjectiveFunctions.goldprice_rescaled_bulk(X)
    return X, y

def getInitialGoldpriceSCDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the rescaled
    Goldstein-Price function (2d).
    Usage:

    [X, y] = getInitialGoldpriceSCDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -2
    upper_x  = 2
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 1, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.goldprice_rescaled_bulk(X)
    return X, y

def getInitialGriewankData(num_points, dim):
    """
    Gets the initial randomly generated data for the GRiewank function (Nd).
    Usage:

    [X, y] = getInitialGriewankData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -5
    upper_x  = 5
    X        = getInitialInputData(num_points, dim, lower_x, upper_x)
    y        = ObjectiveFunctions.griewank_bulk(X)
    return X, y

def getInitialGriewankDataABO(num_points, dim, max_time):
    """
    Gets the initial randomly generated data for the Griewank function
    (Nd).
    Usage:

    [X, y] = getInitialGriewankDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -5
    upper_x  = 5
    dim_     = dim -1
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, dim_, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.griewank_bulk(X)
    return X, y


def getInitialRastriginData(num_points, dim):
    """
    Gets the initial randomly generated data for the Rastrigin function (Nd).
    Usage:

    [X, y] = getInitialRastriginData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           [X, y]:     datapoints generated (number_of_points * 2)
    """
    lower_x  = -5
    upper_x  = 5
    X        = getInitialInputData(num_points, dim, lower_x, upper_x)
    y        = ObjectiveFunctions.rastrigin_bulk(X)
    return X, y

def getInitialRastriginDataABO(num_points, dim, max_time):
    """
    Gets the initial randomly generated data for the Rastrigin function
    (Nd).
    Usage:

    [X, y] = getInitialRastriginDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           dim:        dimensionality
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * N)
    """
    lower_x  = -5
    upper_x  = 5
    dim_     = dim -1
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, dim_, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.rastrigin_bulk(X)
    return X, y

def getInitialHartmann4Data(num_points):
    """
    Gets the initial randomly generated data for the Hartmann4 function (4d).
    Usage:

    [X, y] = getInitialHartmann4Data(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * N)
    """
    lower_x  = 0
    upper_x  = 1
    X        = getInitialInputData(num_points, 4, lower_x, upper_x)
    y        = ObjectiveFunctions.hartmann4_bulk(X)
    return   X, y

def getInitialHartmann4DataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Hartmann4 function (4d).
    Usage:

    [X, y] = getInitialHartmann4DataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 4)
    """
    lower_x  = 0
    upper_x  = 1
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 3, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.hartmann4_bulk(X)
    return   X, y


def getInitialShekelData(num_points):
    """
    Gets the initial randomly generated data for the Shekel function (4d).
    Usage:

    [X, y] = getInitialShekelData(num_points)

           num_points: number of datapoints neeeded (1 x 1)
           [X, y]:     datapoints generated (number_of_points * 4)
    """
    lower_x  = 0
    upper_x  = 10
    X        = getInitialInputData(num_points, 4, lower_x, upper_x)
    y        = ObjectiveFunctions.shekel_bulk(X)
    return   X, y

def getInitialShekelDataABO(num_points, max_time):
    """
    Gets the initial randomly generated data for the Shekel function (4d).
    Usage:

    [X, y] = getInitialShekelDataABO(num_points, max_time)

           num_points: number of datapoints neeeded (1 x 1)
           max_time:   maximum time for initial data
           [X, y]:     datapoints generated (number_of_points * 4)
    """
    lower_x  = 0
    upper_x  = 10
    X1       = getInitialInputData(num_points, 1, lower_x, max_time)
    X2       = getInitialInputData(num_points, 3, lower_x, upper_x)
    X        = numpy.concatenate((X1,X2), axis=1)
    y        = ObjectiveFunctions.shekel_bulk(X)
    return   X, y
