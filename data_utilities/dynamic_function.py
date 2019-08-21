# -*- coding: utf-8 -*-
# 3-dimensional dynamic function from Marchant et al. (2014) for testing Bayesian
# optimisation for dynamic problems in spatio-temporal monitoring. The function is:
#
# y = f(x1,x2,t) = \exp(-\frac{x1-2-f1(t)}{0.7}) * \exp(-\frac{x2-2-f2(t)}{0.7})
#
# where
#   f1(t) = 1.5 * sin(2*pi*t)
#   f2(t) = 1.5 * cos(2*pi*t)
#
# with
#   x1,x2 \in [0,5] and t \in [0, \infty]
#
# Usage:
#   y = dynamic_function(x)
# where
#   x [1x3] vector, and
#   y is a scaler
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com> 27/MAY/2017

import numpy as numpy

def DynamicFunction(x):
    t      = x[0]
    x1     = x[1]
    x2     = x[2]
    y      = dynamic_function_auxillary(t, x1, x2)
    return y

def dynamic_function_auxillary(t, x1, x2):
    part_1 = getExponent1(x1, t)
    part_2 = getExponent2(x2, t)
    y      = numpy.exp(part_1) * numpy.exp(part_2)
    return y

def getExponent1(x, t):
    temp0     = f_1(t)
    temp1     = x - 2 - temp0
    temp2     = .7
    temp3     = (temp1/temp2)**2
    exponent1 = - temp3
    return exponent1

def getExponent2(x, t):
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
