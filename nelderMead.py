# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import sklearn.metrics as mt
from sklearn.svm import SVC
from scipy.optimize import minimize
import numpy as np
from numpy import random as rd
import math as mh


def c_func(a):
    slack = rd.rand(1)
    return 0.5 * (a[0] ** 2) + a[1] * slack


def gamma_func(b):
    return mh.exp(-b[0] * ((b[1] - b[2]) ** 2))


class Optimization:

    def __init__(self, x, y):
        # initiation value x and y from svmPredict.py
        self.X = x
        self.Y = y
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.30)
        # initiation range value for C and Gamma
        self.log_c = np.logspace(0, 3)
        self.log_gamma = np.logspace(0, 2)
        # select random value for nelder-mead algorithm
        self.pt_c = rd.choice(self.log_c, 2)
        self.pt_gamma = rd.choice(self.log_gamma, 3)
        # define C and Gamma Value from find_params() function
        self.c_val, self.gamma_val= np.array(self.find_params())

    def find_params(self):
        # Nelder-Mead Function for find C and Gamma value
        result_c = minimize(c_func, self.pt_c, method='nelder-mead')
        result_gamma = minimize(gamma_func, self.pt_gamma, method='nelder-mead')

        # solution_c = result_c['x']
        # solution_gamma = result_gamma['x']
        # eva_c = c_func(solution_c)
        # eva_gamma = gamma_func(solution_gamma)
        # print('Solution: f(%s)=%.5f' % (solution_c, eva_c))
        # print('Solution: f(%s)=%.5f' % (solution_gamma, eva_gamma))

        # Nelder-Mead Function for find C and Gamma value
        c = float(result_c.x[1])
        gamma = float(result_gamma.x[0])
        return c, gamma

    def nm_func(self):
        cv = StratifiedKFold(n_splits=10)
        svc = SVC(C=self.c_val, gamma=self.gamma_val, kernel='rbf')
        predicted = cross_val_predict(svc, self.X, self.Y, cv=cv)
        print(mt.classification_report(self.Y, predicted))
        print(mt.confusion_matrix(self.Y, predicted))
