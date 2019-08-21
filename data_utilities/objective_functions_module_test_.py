# This script tests the objective functions to ensure that they return correct
# responses.
#
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com> 27/MAY/2017

# import DynamicFunction

import numpy
from   objective_functions import *


x_1                                   = numpy.array([512,404.2319],   float )
x_2                                   = numpy.array([0.0898,-0.7126], float )
x_3                                   = numpy.array([-0.0898,0.7126], float ) 
x_4                                   = numpy.array([-2.903534,-2.903534], float )
x_5                                   = numpy.array([2, -2.9,-2], float ) 
x_6                                   = numpy.array([.5, .9,.5], float )
x_7                                   = numpy.array([.0,.0,.0], float )
x_8                                   = numpy.array([.110,.55,.8], float )
x_9                                   = numpy.array([.20,.15,.476,.275,.3116,.657], float )
x_10                                  = numpy.array([0,0], float )
x_11                                  = numpy.array([-40,-40], float )

print( 'branin_separable      = ' + str( branin_separable(x_1[0],x_1[1] ) ) )
print( 'branin                = ' + str( branin(x_1) ) + '\n' )

print( 'eggholder             = ' + str( eggholder(x_1) ) )
print( 'eggholder_separable   = ' + str( eggholder_separable(x_1[0], x_1[1]) ) + '\n' )

print( 'camel6hump            = ' + str( camel6hump(x_2) ) )
print( 'camel6hump_separable  = ' + str( camel6hump_separable(x_2[0],x_2[1]) ) + '\n' )

print( 'camel6hump            = ' + str( camel6hump(x_3) ) )
print( 'camel6hump_separable  = ' + str( camel6hump_separable( x_3[0], x_3[1]) ) + '\n')

print( 'styblinski_tang       = ' + str( styblinski_tang(x_4) ) + '\n' )

# print( 'DynamicFunction                =' + DynamicFunction.DynamicFunction(x_5) + '\n' )

print( 'hartmann3                      = ' + str( hartmann3(x_8) ) + '\n' )

print( 'hartmann6                      = ' + str( hartmann6(x_9) ) + '\n' )

print( 'hartmann6_rescaled             = ' + str( hartmann6_rescaled(x_9) ) + '\n' )

print( 'ackley                = ' + str( ackley(x_10) ) )
print( 'ackley                = ' + str( ackley(x_11) ) + '\n' )

print( 'branin_modified                = ' + str( branin_modified(x_1) ) )
print( 'branin_modified                = ' + str( branin_modified(x_10) ) )
print( 'branin_modified                = ' + str( branin_modified(x_11) ) + '\n' )

print( 'branin_rescaled                = ' + str( branin_rescaled(x_1) ) )
print( 'branin_rescaled                = ' + str( branin_rescaled(x_10) ) )
print( 'branin_rescaled                = ' + str( branin_rescaled(x_11) ) + '\n' )
