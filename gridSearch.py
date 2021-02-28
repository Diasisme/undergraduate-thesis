from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np


class Optimization:

    def __init__(self, x, y):
        # initiation value x and y from svmPredict.py
        self.X = x
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30)
        # define parameter range
        self.param_grid = {'C': np.logspace(0, 3), 'gamma': np.logspace(0, 3), 'kernel': ['rbf']}

    def grid_func(self):
        # initiation for GridSearch
        grid = GridSearchCV(SVC(), self.param_grid, refit=True, verbose=1, cv=10)
        # fitting model fro GridSearch
        grid.fit(self.X_train, self.y_train)
        # prediction using GridSearch
        grid_predict = grid.predict(self.X_test)
        # print accuracy
        print(classification_report(self.y_test, grid_predict))
