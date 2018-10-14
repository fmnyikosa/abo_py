# This module tests Bayesian Optimization on test functions.
# Function: 2D Styblisnki-Tang 
# Domain:   x_i \in [ -5, 5 ]
#
# Copyright (c) Mandanji Nyikosa, 23/August/2018

# from data_utilities import data_generator      as gen
# from bayes_opt      import abo
# from matplotlib     import pyplot              as plt

from   data_utilities import objective_functions as funcs
from   data_utilities import data_generator      as gen
import GPy      as gpy
import GPyOpt   as gpyopt
import numpy    as np

lb          = -5
ub          =  5

bounds      = [ {'name': 'var_1', 'type': 'continuous', 'domain': ( lb , ub ) },
                {'name': 'var_2', 'type': 'continuous', 'domain': ( lb , ub ) } ] 

max_iter    = 20
my_problem  = gpyopt.methods.BayesianOptimization( 
                f                = funcs.styblinski_tang_bulk, 
                domain           = bounds,
                model_type       = 'GP',
                acquisition_type = 'EI',
                evaluator_type   = 'local_penalization',
                batch_size       = 3,
                num_cores        = 3,
                normalize_Y      = True
            )

my_problem.run_optimization( max_iter )

print("\n===========================")
print("Styblisnki-Tang Function: ")
print( "x_opt = " + str( my_problem.x_opt  )   )
print( "f_opt = " + str( my_problem.fx_opt )   )
print("===========================\n")

# my_problem.plot_acquisition()
# my_problem.plot_convergence()
