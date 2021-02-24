from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.optimize import minimize
import numpy as np

class Optimization:

    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30)


    # def find_params(self):

