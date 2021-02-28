from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
from scipy.optimize import minimize
import numpy as np
from numpy import random as rd
import math as mt


def c_func(a):
    return 0.5 * (a[0] ** 2) + a[1] * 1


def gamma_func(b):
    return mt.exp(-b[0] * ((b[1] - b[2]) ** 2))


class Optimization:

    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.30)
        self.logNum = np.logspace(0, 3)
        self.pt_c = rd.choice(self.logNum, 2)
        self.pt_gamma = rd.choice(self.logNum, 3)

    def find_params(self):
        result_c = minimize(c_func, self.pt_c, method='nelder-mead')
        result_gamma = minimize(gamma_func, self.pt_gamma, method='nelder-mead')
        solution_c = result_c['x']
        solution_gamma = result_gamma['x']
        eva_c = c_func(solution_c)
        eva_gamma = gamma_func(solution_gamma)
        print('Solution: f(%s)=%.5f' % (solution_c, eva_c))
        print('Solution: f(%s)=%.5f' % (solution_gamma, eva_gamma))

