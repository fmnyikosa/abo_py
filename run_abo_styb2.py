# This module tests Bayesian Optimization on test functions.
# Function: 2D Styblisnki-Tang 
# Domain:   x_i \in [ -5, 5 ]
#
# Copyright (c) Mandanji Nyikosa, 23/August/2018

from data_utilities import objective_functions as funcs
from data_utilities import data_generator      as gen

import GPy      as gpy
import GPyOpt   as gpyopt
import numpy    as np
import time

tic          = time.time()

print("============================")
print("Styblisnki-Tang Function: \n")
name        = "styb"
fx           = funcs.styblinski_tang_bulk ## f(x)
lower        = -5
upper        =  5
bounds       = [ {'name': 'var_1', 'type': 'continuous', 'domain': ( lower , upper ) },
                 {'name': 'var_2', 'type': 'continuous', 'domain': ( lower , upper ) } ] 

max_iter     = 90              ## maximum number of iterations for actual BO
num_init_pts = 10              ## number of initial datapoints to generate via LHS
t_span       = upper - lower   ## maximum number of iterations
t_delta      = float( t_span / (max_iter + num_init_pts) )   ## time delta
t_max        = upper           ## maximum allowed value for the temporal variable

t_init       = lower + ( num_init_pts * t_delta )
X, Y         = gen.getInitialStybTangDataABO( num_init_pts , t_init )

my_problem   = gpyopt.methods.BayesianOptimization( 
                f                = fx, 
                domain           = bounds,
                X                = X,
                Y                = Y,
                model_type       = 'GP',
                acquisition_type = 'EI',
                evaluator_type   = 'sequential', #'local_penalization',
                batch_size       = 1,
                # num_cores        = 3,
                normalize_Y      = True
            )

max_time     = 60   ## maximum allowed time ( seconds )
eps          = 0    ## tolerance, max distance between consecutive evaluations

# do loop
t_current = t_init
for i in range( max_iter + 1 ):
    print( "i       = "  + str( i ) )
    print( "t       = "  + str( t_current ) )
    my_problem.run_optimization(
        1,
        eps              = eps,
        context          = { 'var_1': t_current },
        report_file      = "abo_log/report/"+name+"_report_"+str(i)+".txt", 
        evaluations_file = "abo_log/evals/" +name+"_evals_" +str(i)+".csv", 
        models_file      = "abo_log/model/" +name+"_models_"+str(i)+".csv"
        )
    t_current = t_current + t_delta
    # print( "t_init  = "  + str( t_init ) )
    # print( "t_delta = "  + str( t_delta ) )
    # print( "t_max   = "  + str( t_max ) )
    print( "x_opt   = "  + str( my_problem.X[-1,:]  )   )
    print( "f_opt   = "  + str( my_problem.Y[-1,:] )  + "\n\n" )

print("============================")

toc          = time.time()
elapsed      = toc - tic

print(  "The experiment took %f seconds to run."%(elapsed) )

my_problem.plot_acquisition()
# my_problem.plot_convergence()