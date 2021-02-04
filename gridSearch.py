import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class GridSearch:

    def __init__(self, x, y):
        # initiation value x and y from svmPredict.py
        self.X = x
        self.y = y
        # split dataset to train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)
        # define parameter range
        self.param_grid = {'C': [0.1, 1, 10, 100, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

    def grid_func(self):
        # initiation for GridSearch
        grid = GridSearchCV(SVC(), self.param_grid, refit=True, verbose=3)
        # fitting model fro GridSearch
        grid.fit(self.X_train, self.y_train)
        # prediction using GridSearch
        grid_predict = grid.predict(self.X_test)
        # print accuracy
        print(classification_report(self.y_test, grid_predict))
