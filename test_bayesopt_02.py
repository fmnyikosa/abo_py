# This module tests Bayesian Optimization on test functions.
# Function: 2D 6-Hump Camelback 
# Domain:   x_i \in [ -5, 5 ]
#
# Copyright (c) Mandanji Nyikosa, 23/August/2018

# from data_utilities import data_generator      as gen
# from bayes_opt      import abo
# from matplotlib     import pyplot              as plt 
from data_utilities import objective_functions as funcs

import GPy      as gpy
import GPyOpt   as gpyopt
import numpy    as np
import sklearn  as skl

lb1         = -3
ub1         =  3

lb2         = -2
ub2         =  2

bounds      = [ {'name': 'var_1', 'type': 'continuous', 'domain': (  lb1 , ub1 ) },
                {'name': 'var_2', 'type': 'continuous', 'domain': (  lb2 , ub2 ) } ] 

max_iter    = 100
my_problem  = gpyopt.methods.BayesianOptimization( funcs.camel6hump_bulk, bounds )

my_problem.run_optimization( max_iter )

print("6-Hump Camelback Function: \n")
print( "x_opt = " + str( my_problem.x_opt  )   )
print( "f_opt = " + str( my_problem.fx_opt )   + "\n"  )

# my_problem.plot_acquisition()
my_problem.plot_convergence()
