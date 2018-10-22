import GPy
import GPyOpt
import matplotlib

from   numpy.random   import seed
# from   data_utilities import objective_functions

def myf(x):
    return (2*x)**2

bounds      = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]
max_iter    = 15
my_problem  = GPyOpt.methods.BayesianOptimization( myf , bounds )

my_problem.run_optimization( max_iter )

print("x_opt = " + str( my_problem.x_opt) )
print("f_opt = " + str( my_problem.fx_opt ) )

my_problem.plot_acquisition()
my_problem.plot_convergence()



