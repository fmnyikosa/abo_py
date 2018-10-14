# This module tests Bayesian Optimization on test functions.
# Function: 6D Hartmann 
# Domain:   x_i \in [ -1, 1 ]
#
# Copyright (c) Mandanji Nyikosa, 23/August/2018

from data_utilities import objective_functions as funcs
# from data_utilities import data_generator      as gen
from matplotlib     import pyplot              as plt 
# from bayes_opt      import abo

import GPy      as gpy
import GPyOpt   as gpyopt
import numpy    as np
import sklearn  as skl

lb          = -1
ub          =  1

bounds      = [ {'name': 'var_1', 'type': 'continuous', 'domain': (  lb , ub ) },
                {'name': 'var_2', 'type': 'continuous', 'domain': (  lb , ub ) },
                {'name': 'var_3', 'type': 'continuous', 'domain': (  lb , ub ) },
                {'name': 'var_4', 'type': 'continuous', 'domain': (  lb , ub ) },
                {'name': 'var_5', 'type': 'continuous', 'domain': (  lb , ub ) },
                {'name': 'var_6', 'type': 'continuous', 'domain': (  lb , ub ) } ] 

max_iter    = 100
my_problem  = gpyopt.methods.BayesianOptimization( funcs.hartmann6_bulk, bounds )

my_problem.run_optimization( max_iter )

print("Hartmann 6D Function: ")
print( "x_opt = " + str( my_problem.x_opt)   )
print( "f_opt = " + str( my_problem.fx_opt )   + "\n"  )

# my_problem.plot_acquisition()
my_problem.plot_convergence()
