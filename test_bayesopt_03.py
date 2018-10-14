# This module tests Bayesian Optimization on test functions.
# Function: 2D Branin  
# Domain:   x_i \in [ -5, 5 ]
#
# Copyright (c) Mandanji Nyikosa, 23/August/2018

# from data_utilities import data_generator      as gen
# from bayes_opt      import abo
from data_utilities import objective_functions as funcs
from matplotlib     import pyplot              as plt 


import GPy      as gpy
import GPyOpt   as gpyopt
import numpy    as np
import sklearn  as skl

lb1         = -5
ub1         =  10

lb2         =  0
ub2         =  15

bounds      = [ {'name': 'var_1', 'type': 'continuous', 'domain': (  lb1 , ub1 ) },
                {'name': 'var_2', 'type': 'continuous', 'domain': (  lb2 , ub2 ) } ] 

max_iter    = 100
my_problem  = gpyopt.methods.BayesianOptimization( funcs.branin_bulk, bounds )

my_problem.run_optimization( max_iter )

print("Branin Function: ")
print( "x_opt = " + str( my_problem.x_opt  )   )
print( "f_opt = " + str( my_problem.fx_opt )   + "\n"  )

# my_problem.plot_acquisition()
my_problem.plot_convergence()
