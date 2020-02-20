import numpy as np
import GPyOpt

domain = [{'hidden_units'   : 'var_1', 'type': 'discrete',   'domain': tuple(np.arange(1000,4000,100, dtype = np.int))},
          {'p'              : 'var_2', 'type': 'continous',  'domain': (0 , 1)},
          {'activation_func': 'var_3', 'type': 'categorical','domain': tuple(np.arange(5))}]
activ_func = [{ 0 : 'tanh'}, 
			  { 1 : 'ReLU'},
			  { 2 : 'ReLU6'},
			  { 3 : 'sigmoid'},
			  { 4 : 'Linear'}]

opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = 'EI',      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=.1
opt.run_optimization(max_iter = 5)
x_best = opt.X[np.argmin(opt.Y)]
print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(x_best[1]) + ", max_features=" + str(
    x_best[2])  + ", criterion=" + str(
    x_best[3]))