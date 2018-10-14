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
fx           = funcs.styblinski_tang_bulk ## f(x)
lower        = -5
upper        =  5
bounds       = [ {'name': 'var_1', 'type': 'continuous', 'domain': ( lower , upper ) },
                 {'name': 'var_2', 'type': 'continuous', 'domain': ( lower , upper ) } ] 

max_iter     = 9               ## maximum number of iterations for actual BO
num_init_pts = 1               ## number of initial datapoints to generate via LHS
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
                evaluator_type   = 'local_penalization',
                batch_size       = 3,
                num_cores        = 3,
                normalize_Y      = True
            )

max_time     = 60   ## maximum allowed time ( seconds )
eps          = 0    ## tolerance, max distance between consecutive evaluations

print(X)
print(Y)

# do loop
t_current = t_init
for i in range(max_iter):
    my_problem.run_optimization(
        1,
        eps              = eps,
        context          = { 'var_1': t_current },
        report_file      = "report", 
        evaluations_file = "evals", 
        models_file      = "models"
        )
    t_current = t_current + t_delta
    print( "i       = "  + str( i ) )
    print( "t       = "  + str( t_current ) )
    print( "t_init  = "  + str( t_init ) )
    print( "t_delta = "  + str( t_delta ) )
    print( "t_max   = "  + str( t_max ) )
    print( "x_opt   = "  + str( my_problem.X[-1,:]  )   )
    print( "f_opt   = "  + str( my_problem.Y[-1,:] )  + "\n\n" )

print("============================")

toc          = time.time()
elapsed      = toc - tic

print(  "The experiment took %f seconds to run."%(elapsed) )
print(  "The experiment took {:f} seconds to run.".format( elapsed )  )
print( f"The experiment took {elapsed:f} seconds to run." )

my_problem.plot_acquisition()
# my_problem.plot_convergence()










#======================================= Code Graveyard ==================================

# my_problem.plot_acquisition()
# my_problem.plot_convergence()

# my_problem = gpyopt.methods.BayesianOptimization(f=func.f,                 # Objective function       
#                                              domain=mixed_domain,          # Box-constraints of the problem
#                                              initial_design_numdata = 5,   # Number data initial design
#                                              acquisition_type='EI',        # Expected Improvement
#                                              model_type= 'GP_MCMC',
#                                              exact_feval = True,
#                                              evaluator_type = 'local_penalization',
#                                              batch_size = 5
#                                              )                             # True evaluations, no sample noise

# mixed_domain =[{'name': 'var1', 'type': 'continuous',  'domain': (-5,5), 'dimensionality': 3 },
#                {'name': 'var2', 'type': 'discrete',    'domain': (3,8,10)},
#                {'name': 'var3', 'type': 'categorical', 'domain': (0,1,2)},
#                {'name': 'var4', 'type': 'continuous',  'domain': (-1,2)} ]

# myBopt.run_optimization(max_iter,eps=eps)
# myBopt.run_optimization(max_iter,eps=eps,context = {'var1_1':.3, 'var1_2':0.4})
# myBopt.run_optimization(max_iter,eps=eps,context = {'var1_1':0, 'var3':2})
# myBopt.run_optimization(max_iter,eps=eps,context = {'var1_1':0, 'var2':3},)
# myBopt.run_optimization(max_iter,eps=eps,context = {'var1_1':0.3, 'var3':1, 'var4':-.4})
# myBopt.run_optimization(max_iter,eps=eps)

# constraints = [{'name': 'constr_1', 'constraint': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
#               {'name': 'constr_2', 'constraint': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}]

# func  = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5)

# X           = np.array( 
#                         [
#                             [ -5.0,  1 ],
#                             [ -4.9, -3 ],
#                             [ -4.8,  5 ],
#                             [ -4.7,  0 ],
#                             [ -4.6,  5 ]
#                         ], float
#                       )
# Y           = np.array( 
#                         [
#                             f_([ -5.0,  1 ]),
#                             f_([ -4.9, -3 ]),
#                             f_([ -4.8,  5 ]),
#                             f_([ -4.7,  0 ]),
#                             f_([ -4.6,  5 ])
#                         ], float
#                       )
# print( "x_opt = " + str( my_problem.x_opt  )   )
# print( "f_opt = " + str( my_problem.fx_opt )  + "\n" )

# np.round(myBopt.X,2)

# my_problem.run_optimization( max_iter )



