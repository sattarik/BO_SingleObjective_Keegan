'''
3D DLP printing problem suite.
'''

import numpy as np
from pymoo.factory import get_reference_directions
from pymoo.problems.util import load_pareto_front_from_file

from autooed.problem.problem import Problem

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class laser(Problem):

    config = {
        'type': 'integer',
        'n_var': 4,
        'n_obj': 1,
        'n_constr': 0,
        'var_lb': [0, 0, 0, 0],
        'var_ub': [100, 100, 100, 100]
    }

    def __init__(self):
        super().__init__()
        self.k = self.n_var - self.n_obj + 1
    
    def obj_func(self, x_):
        f = []

        for i in range(0, self.n_obj):
            _f = float (input (
            "ratios A-B {} sum {} Enter objective {}: ".
             format(np.round(x, 4), np.sum(np.round(x, 4)), i)))
            _f_ = -_f
            f.append(_f)
          

        f = np.array(f)
        return f

    def evaluate_objective(self, x_, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            
            _f = float (input (
            "ratios A-B {} sum {} Enter objective {}: ".
             format(np.round(x_, 4), np.sum(np.round(x_, 4)), i)))
            _f = -_f
            f.append(_f)

        f = np.array(f)
        return f
    

class contact_angle(laser):
    def _calc_pareto_front(self):
        ref_kwargs = dict(n_points=100) if self.n_obj == 1 else dict(n_partitions=15)
        ref_dirs = get_reference_directions('das-dennis', n_dim=self.n_obj, **ref_kwargs)
        return 0.5 * ref_dirs
   
    def evaluate_objective(self, x_, alpha=1):
        f = []
        objectives = ['Contact angle']
        for i in range(0, self.n_obj):
            while True:
                try:
                    _f = float (input (
                            "ratios A-B {} sum {} Enter objective {}: ".
                            format(np.round(x_, 2), np.sum(np.round(x_, 2)), objectives[i])))
                except ValueError:
                    print ('the objective {} was not valid, try again'.format(objectives[i]))
                    continue
                else:
                    break
            _f = -_f
            f.append(_f)

        f = np.array(f)
        return f

