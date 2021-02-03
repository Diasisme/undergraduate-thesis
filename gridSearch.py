import pandas as pd
import numpy as np
import sklearn.svm as svc


class GridSearch:

    def __init__(self, x, y):
        self.X = x
        self.y = y

    def grid_func(self):
        print(self.X)
