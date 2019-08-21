# -*- coding: utf-8 -*-
# This module contains all the test objective function for Bayesian optimisation
# The list of test objective functins include:
#   6-Hump Camelback function
#   Ackley ND function
#   Branin 2D function
#   (Modified) Branin 2D function
#   (Rescaled) Branin 2D function
#   Colville 4D function
#   Eggholder 2D function
#   Goldstein-Price ND function
#   (Rescaled) Goldstein-Price function
#   Griewank 4D function
#   Hartmann 3D function
#   Hartmann 4D function
#   Hartmann 6D function
#   (Rescaled) Hartmann 6D function
#   Rastrigin ND function
#   Shekel ND function
#   Styblinski-Tang 2D function
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com> 27-MAY-2017
# 
# Updated on 24-Aug-2018 @mandanji

import numpy as numpy

def ackley(x):
    dim    = x.size
    a      = 20
    b      = 0.2
    c      = 2 * numpy.pi
    sum1   = 0
    sum2   = 0
    for i in range(0,dim):
        x_i     =  x[i]
        sum1    += x_i**2
        sum2    += numpy.cos( c * x_i )
    term1  = -a * numpy.exp( -b * numpy.sqrt( sum1/dim ) )
    term2  = -numpy.exp( sum2 / dim )
    y      = term1 + term2 + a + numpy.exp(1)
    return y

def ackley_separable(x,y):
    dim    = 2
    a      = 20
    b      = 0.2
    c      = 2 * numpy.pi
    sum1   = 0
    sum2   = 0
    sum1   += x**2
    sum2   += numpy.cos( c * x )
    sum1   += y**2
    sum2   += numpy.cos( c * y )
    term1  = -a * numpy.exp( -b * numpy.sqrt( sum1/dim ) )
    term2  = -numpy.exp( sum2 / dim )
    y      = term1 + term2 + a + numpy.exp(1)
    return y

def branin(x):
    x1     = x[0]
    x2     = x[1]
    a      = 1.0
    b      = 5.1 / (4.*numpy.pi**2)
    c      = 5.0 / numpy.pi
    r      = 6.0
    s      = 10.0
    t      = 1.0 / (8.0 * numpy.pi)
    y      = a*(x2 - b*(x1**2) + c*x1 - r)**2 + s*(1-t)*numpy.cos(x1) + s
    return y

def branin_separable(x1, x2):
    a     = 1.0
    b     = (5.1)/(4.0 * numpy.pi**2)
    c     = 5.0  / numpy.pi
    r     = 6.0
    s     = 10.0
    t     = 1.0  / (8.0 * numpy.pi)
    y     = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t) * numpy.cos(x1) + s
    return y

def branin_bulk(x):
    x1     = x[:,0:1]
    x2     = x[:,1:2]
    a      = 1.
    b      = 5.1 / (4.*numpy.pi**2)
    c      = 5. / numpy.pi
    r      = 6.
    s      = 10.
    t      = 1. / (8.*numpy.pi)
    y      = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*numpy.cos(x1) + s
    return y

def branin_modified(x):
    x1    = x[0]
    x2    = x[1]
    t     = 1 / (8*numpy.pi)
    s     = 10
    r     = 6
    c     = 5/numpy.pi
    b     = 5.1 / (4*numpy.pi**2)
    a     = 1
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s * (1-t) * numpy.cos(x1)
    y     = term1 + term2 + s + 5*x1
    return y

def branin_rescaled(x):
    x1    = x[0]
    x2    = x[1]
    x1_   = (15*x1) - 5
    x2_   = (15 * x2)
    temp0 = (4*(numpy.pi)**2)
    temp1 = (x1_**2)/temp0
    temp2 = (5*x1_ / (numpy.pi)) - 6
    term1 = x2_ - (5.1 * temp1) + temp2
    term2 = ( 10 - 10/(8*numpy.pi)) * numpy.cos(x1_)
    y     = (term1**2 + term2 - 44.81) / 51.95
    return y

def branin_modified_bulk(x):
    x1    = x[:,0:1]
    x2    = x[:,1:2]
    t     = 1 / (8*numpy.pi)
    s     = 10
    r     = 6
    c     = 5/numpy.pi
    b     = 5.1 / (4*numpy.pi**2)
    a     = 1
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s * (1-t) * numpy.cos(x1)
    y     = term1 + term2 + s + 5*x1
    return y

def branin_rescaled_bulk(x):
    x1    = x[:,0:1]
    x2    = x[:,1:2]
    x1_   = (15*x1) - 5
    x2_   = (15 * x2)
    temp0 = (4*(numpy.pi)**2)
    temp1 = (x1_**2)/temp0
    temp2 = (5*x1_ / (numpy.pi)) - 6
    term1 = x2_ - (5.1 * temp1) + temp2
    term2 = ( 10 - 10/(8*(numpy.pi)) ) * numpy.cos(x1_)
    y     = (term1**2 + term2 - 44.81) / 51.95
    return y

def camel6hump(x):
    x1     = x[0]
    x2     = x[1]
    first  = (4. - 2.1 * x1**2 + (x1**4)/3.) * x1**2
    second = x1 * x2
    third  = (-4. + 4. * x2**2) * x2**2
    y      = first + second + third
    return y

def camel6hump_separable(x1, x2):
    first  = (4. - 2.1 * x1**2 + (x1**4)/3.) * x1**2
    second = x1 * x2
    third  = (-4. + 4. * x2**2) * x2**2
    y      = first + second + third
    return y

def camel6hump_bulk(x):
    x1     = x[:,0:1]
    x2     = x[:,1:2]
    first  = (4. - 2.1 * x1**2 + (x1**4)/3.) * x1**2
    second = x1 * x2
    third  = (-4. + 4. * x2**2) * x2**2
    y      = first + second + third
    return y

def eggholder(x):
    x1     = x[0]
    x2     = x[1]
    first  = -(x2 + 47) * numpy.sin( numpy.sqrt( numpy.absolute(x2 + 0.5*x1 + 47) ) )
    second = x1*numpy.sin(numpy.sqrt(numpy.absolute(x1 - (x2 + 47))))
    y      = first - second
    return y

def eggholder_separable(x1, x2):
    first  = -(x2 + 47) * numpy.sin( numpy.sqrt( numpy.absolute(x2 + 0.5*x1 + 47) ) )
    second = x1*numpy.sin(numpy.sqrt(numpy.absolute(x1 - (x2 + 47))))
    y      = first - second
    return y

def eggholder_bulk(x):
    x1     = x[:,0:1]
    x2     = x[:,1:2]
    first  = -(x2 + 47)*numpy.sin( numpy.sqrt( numpy.absolute(x2 + 0.5*x1 +47)))
    second = x1*numpy.sin(numpy.sqrt(numpy.absolute(x1 - (x2 + 47))))
    y      = first - second
    return y

def hartmann3(x):
    alpha = numpy.array([1.0,1.2,3.0,3.2], float)
    A     = numpy.array([[3.0,10,30],\
                         [0.1,10,35],\
                         [3.0,10,30],\
                         [0.1,10,35]],float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[3689,1170,2673],\
                         [4699,4387,7470],\
                         [1091,8732,5547],\
                         [381,5743,8828]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,3):
            xj    = x[j]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new    = alpha[i] * numpy.exp(-inner)
        outer  = outer + new
    y = -outer
    return y

def hartmann3_bulk(x):
    alpha = numpy.array([1.0,1.2,3.0,3.2], float)
    A     = numpy.array([[3.0,10,30],\
                         [0.1,10,35],\
                         [3.0,10,30],\
                         [0.1,10,35]],float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[3689,1170,2673],\
                         [4699,4387,7470],\
                         [1091,8732,5547],\
                         [381,5743,8828]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,3):
            xj    = x[:,j:j+1]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new    = alpha[i] * numpy.exp(-inner)
        outer  = outer + new
    y = -outer
    return y

def hartmann6(x):
    alpha = numpy.array( [1.0,  1.2, 3.0,  3.2], float)
    A     = numpy.array([[10,   3,   17,   3.5, 1.7, 8],\
                         [0.05, 10,  17,   0.1, 8,   14],\
                         [3,    3.5, 1.7,  10,  17,  8],\
                         [17,   8,   0.05, 10,  0.1, 14]], float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[1312, 1696, 5569, 124,  8283, 5886],\
                         [2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],\
                         [4047, 8828, 8732, 5743, 1091, 381]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,6):
            xj    = x[j]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new    = alpha[i] * numpy.exp(-inner)
        outer  = outer + new
    y = -(2.58 + outer) / 1.94
    return y

def hartmann6_bulk(x):
    alpha = numpy.array( [1.0,  1.2, 3.0,  3.2], float)
    A     = numpy.array([[10,   3,   17,   3.5, 1.7, 8],\
                         [0.05, 10,  17,   0.1, 8,   14],\
                         [3,    3.5, 1.7,  10,  17,  8],\
                         [17,   8,   0.05, 10,  0.1, 14]], float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[1312, 1696, 5569, 124,  8283, 5886],\
                         [2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],\
                         [4047, 8828, 8732, 5743, 1091, 381]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,6):
            xj    = x[:,j:j+1]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new    = alpha[i] * numpy.exp(-inner)
        outer  = outer + new
    y = -(2.58 + outer) / 1.94
    return y

def hartmann6_rescaled(x):
    alpha = numpy.array( [1.0,  1.2, 3.0,  3.2], float)
    A     = numpy.array([[10,   3,   17,   3.5, 1.7, 8],\
                         [0.05, 10,  17,   0.1, 8,   14],\
                         [3,    3.5, 1.7,  10,  17,  8],\
                         [17,   8,   0.05, 10,  0.1, 14]], float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[1312, 1696, 5569, 124,  8283, 5886],\
                         [2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],\
                         [4047, 8828, 8732, 5743, 1091, 381]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,6):
            xj    = x[j]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new   = alpha[i] * numpy.exp(-inner)
        outer = outer + new
    y     = -outer
    return y

def hartmann6_rescaled_bulk(x):
    alpha = numpy.array( [1.0,  1.2, 3.0,  3.2], float)
    A     = numpy.array([[10,   3,   17,   3.5, 1.7, 8],\
                         [0.05, 10,  17,   0.1, 8,   14],\
                         [3,    3.5, 1.7,  10,  17,  8],\
                         [17,   8,   0.05, 10,  0.1, 14]], float)
    temp0 = 10**(-4)
    temp1 = numpy.array([[1312, 1696, 5569, 124,  8283, 5886],\
                         [2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],\
                         [4047, 8828, 8732, 5743, 1091, 381]],float)
    P     = temp0 * temp1
    outer = 0
    for i in range(0,4):
        inner  = 0
        for j in range(0,6):
            xj    = x[:,j:j+1]
            Aij   = A[i,j]
            Pij   = P[i,j]
            tempj = (xj-Pij)**2
            inner = inner + (Aij*tempj)
        new   = alpha[i] * numpy.exp(-inner)
        outer = outer + new
    y = -outer
    return y

def styblinski_tang(x):
    dimensions = numpy.size(x)
    temp       = 0
    for index in range(0,dimensions):
        x_i    = x[index]
        temp   = temp + x_i**4 - 16*x_i**2 + 5*x_i
    y = temp * 0.5
    return y

def styblinski_tang_bulk(x):
    dimensions = x.shape[1]
    temp       = 0
    for index in range(0,dimensions):
        x_i    = x[:, index:index+1]
        temp   = temp + x_i**4 - 16*x_i**2 + 5*x_i
    y = temp * 0.5
    return y

def styblinski_tang_separable(x,y):
    temp   = 0
    temp   = temp + x**4 - 16*x**2 + 5*x
    temp   = temp + y**4 - 16*y**2 + 5*y
    y      = temp * 0.5
    return y

#-------------------------------------------------------------------------------
def dynamic(x):
    """
    3-dimensional dynamic function from Marchant et al. (2014) for testing Bayesian
    optimisation for dynamic problems in spatio-temporal monitoring. The function is:

    y = f(x1,x2,t) = \exp(-\frac{x1-2-f1(t)}{0.7}) * \exp(-\frac{x2-2-f2(t)}{0.7})

    where:
      f1(t) = 1.5 * sin(2*pi*t)
      f2(t) = 1.5 * cos(2*pi*t)

    with:
      x1,x2 \in [0,5] and t \in [0, \infty]

    Usage:
       y = dynamic(x)
    where
       x [1x3] vector, and
       y is a scaler
    """
    t         = x[:,0:1]
    x1        = x[:,1:2]
    x2        = x[:,2:3]
    y         = dynamic_auxillary(t, x1, x2)
    return y

def dynamic_auxillary(t, x1, x2):
    part_1    = get_exponent1(x1, t)
    part_2    = get_exponent2(x2, t)
    y         = numpy.exp(part_1) * numpy.exp(part_2)
    return y

def get_exponent1(x, t):
    temp0     = f_1(t)
    temp1     = x - 2 - temp0
    temp2     = .7
    temp3     = (temp1/temp2)**2
    exponent1 = - temp3
    return exponent1

def get_exponent2(x, t):
    temp0     = f_2(t)
    temp1     = x - 2 - temp0
    temp2     = .7
    temp3     = (temp1/temp2)**2
    exponent2 = - temp3
    return exponent2

def f_1(t):
    f1        = 1.5 * numpy.sin(2 * numpy.pi * t)
    return f1

def f_2(t):
    f2        = 1.5 * numpy.cos(2 * numpy.pi * t)
    return f2


#-------------------------------------------------------------------------------
#                        Added later: Oct 15th 2017
#-------------------------------------------------------------------------------

def colville_bulk(xx):
    x1     = xx[:,0:1]
    x2     = xx[:,1:2]
    x3     = xx[:,2:3]
    x4     = xx[:,3:4]
    term1  = 100 * (x1**2-x2)**2
    term2  = (x1-1)**2
    term3  = (x3-1)**2
    term4  = 90 * (x3**2-x4)**2
    term5  = 10.1 * ((x2-1)**2 + (x4-1)**2)
    term6  = 19.8*(x2-1)*(x4-1)
    y      = term1 + term2 + term3 + term4 + term5 + term6
    return y

def goldprice_bulk(xx):
    x1     = xx[:,0:1]
    x2     = xx[:,1:2]
    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1  = 1 + fact1a*fact1b
    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2  = 30 + fact2a*fact2b
    y      = fact1*fact2
    return y

def goldprice_separable(x,y):
    x1     = x
    x2     = y
    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1  = 1 + fact1a*fact1b
    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2  = 30 + fact2a*fact2b
    y      = fact1*fact2
    return y

def goldprice_rescaled_bulk(xx):
    x1bar  = 4*xx[:,0:1] - 2
    x2bar  = 4*xx[:,1:2] - 2
    fact1a = (x1bar + x2bar + 1)**2
    fact1b = 19 - 14*x1bar + 3*x1bar**2 - 14*x2bar + 6*x1bar*x2bar + 3*x2bar**2
    fact1  = 1 + fact1a * fact1b
    fact2a = (2*x1bar - 3*x2bar)**2
    fact2b = 18 - 32*x1bar + 12*x1bar**2 + 48*x2bar -36*x1bar*x2bar+27*x2bar**2
    fact2  = 30 + fact2a * fact2b
    prod   = fact1 * fact2
    y      = ( numpy.log(prod) - 8.693 ) / 2.427
    return y

def goldprice_rescaled_separable(x,y):
    x1bar  = x
    x2bar  = y
    fact1a = (x1bar + x2bar + 1)**2
    fact1b = 19 - 14*x1bar + 3*x1bar**2 - 14*x2bar + 6*x1bar*x2bar + 3*x2bar**2
    fact1  = 1 + fact1a * fact1b
    fact2a = (2*x1bar - 3*x2bar)**2
    fact2b = 18 - 32*x1bar + 12*x1bar**2 + 48*x2bar -36*x1bar*x2bar+27*x2bar**2
    fact2  = 30 + fact2a * fact2b
    prod   = fact1 * fact2
    y      = ( numpy.log(prod) - 8.693 ) / 2.427
    return y

def griewank_bulk(xx):
    dim   = xx.shape[1]
    sum   = 0
    prod  = 1
    for ii in range(0, dim):
    	xi   = xx[:,ii:ii+1]
    	sum  = sum + xi**2/4000
    	prod = prod * numpy.cos( xi / numpy.sqrt(ii) )
    y     = sum - prod + 1

def griewank_separable( x , y ):
    sum   = 0
    prod  = 1
    sum   = sum + (x**2) / 4000
    prod  = prod * numpy.cos( x / numpy.sqrt(ii) )
    sum   = sum + (y**2)/4000
    prod  = prod * numpy.cos( y / numpy.sqrt(ii) )
    y     = sum - prod + 1

def hartmann4_bulk(xx):
    alpha = numpy.array( [1.0, 1.2, 3.0, 3.2], float)
    A     = numpy.array([[10, 3, 17, 3.5, 1.7, 8],\
                         [0.05, 10, 17, 0.1, 8, 14],\
                         [3, 3.5, 1.7, 10, 17, 8],\
                         [17, 8, 0.05, 10, 0.1, 14]], float)
    temp0 = 10**(-4)
    temp0 = numpy.array([[1312, 1696, 5569, 124, 8283, 5886],\
                         [2329, 4135, 8307, 3736, 1004, 9991],\
                         [2348, 1451, 3522, 2883, 3047, 6650],\
                         [4047, 8828, 8732, 5743, 1091, 381]], float)
    P     = temp0 * temp1
    outer = 0
    for ii in range(0,4):
    	inner = 0
    	for jj in range(0,4):
    		xj    = xx[:,jj:jj+1]
    		Aij   = A[ii, jj]
    		Pij   = P[ii, jj]
    		inner = inner + Aij*(xj-Pij)**2
    	new   = alpha[ii] * numpy.exp(-inner)
    	outer = outer + new
    y     = (1.1 - outer) / 0.839
    return y

def rastrigin_bulk(xx):
    dim = xx.shape[1]
    sum = 0
    for ii in range(0,dim):
    	xi  = xx[:,ii:ii+1]
    	sum = sum + (xi**2 - 10 * numpy.cos(2 * numpy.pi * xi))
    y   = 10*dim + sum
    return y

def rastrigin_separable(x,y):
    dim = 2
    sum = 0
    sum = sum + (x**2 - 10 * numpy.cos(2 * numpy.pi * x))
    sum = sum + (y**2 - 10 * numpy.cos(2 * numpy.pi * y))
    y   = 10*dim + sum
    return y

def shekel_bulk(x):
    m     = 10
    temp0 = 0.1
    temp1 = numpy.array( [1, 2, 2, 4, 4, 6, 3, 7, 5, 5], float)
    b     = temp0 * temp1
    C     = numpy.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],  \
                         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],  \
                         [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],  \
                         [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]], \
                         float)
    outer = 0
    for ii in range(0,m):
    	b_i    = b[ii]
    	inner  = 0
    	for jj in range(0,4):
    		x_j   = x[:,jj:jj+1]
    		C_ji  = C[jj, ii]
    		inner = inner + (x_j-C_ji)**2
    	outer = outer + 1/(inner+b_i)
    y     = - outer
    return y

#-------------------------------------------------------------------------------
