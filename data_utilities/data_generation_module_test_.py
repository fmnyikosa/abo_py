# -*- coding: utf-8 -*-

from data_generator import *

num_points = 10
dim        = 3
lower_b    = 0
upper_b    = 5

X = getInitialInputData(num_points, dim, lower_b, upper_b)
print( 'Initial data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )

X, y = getInitialBraninData(num_points)
print( 'Branin data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialBraninModData(num_points)
print( 'BraninMod data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialBraninSCData(num_points)
print( 'BraninSC data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialCamel6Data(num_points)
print( 'Camel6 data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialEggholderData(num_points)
print( 'Eggholder data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialHartmann3Data(num_points)
print( 'Hartmann3 data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialHartmann6Data(num_points)
print( 'Hartmann6 data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialHartmann6SCData(num_points)
print( 'Hartmann6SC data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialStybTangData(num_points)
print( 'Styblinski-Tang data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

max_time = 0.5;
X, y = getInitialBraninDataABO(num_points, max_time)
print( 'Branin ABO data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialBraninModDataABO(num_points, max_time)
print( 'BraninMod ABO data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialBraninSCDataABO(num_points, max_time)
print( 'BraninSC ABO data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )

X, y = getInitialCamel6DataABO(num_points, max_time)
print( 'Camel6 ABO data: ' +  '\n' )
print( 'X = \n ' +  str(X)  +  '\n' )
print( 'y = \n ' +  str(y) +   '\n' )
