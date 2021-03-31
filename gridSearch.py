from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as mt
from sklearn.svm import SVC
import numpy as np


class Optimization:

    def __init__(self, x, y):
        # initiation value x and y from svmPredict.py
        self.X = x
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.3)
        # define parameter range
        self.param_grid = {'C': np.logspace(0, 3), 'gamma': np.logspace(0, 2), 'kernel': ['rbf']}
        self.cv = StratifiedKFold(n_splits=10)

    def grid_func(self):
        # initiation for GridSearch
        grid = GridSearchCV(SVC(), self.param_grid, refit=True, verbose=1, cv=self.cv)
        # fitting model for GridSearch
        grid.fit(self.X_train, self.y_train)
        # prediction using GridSearch
        grid_predict = grid.predict(self.X_test)
        # print accuracy
        print(mt.classification_report(self.y_test, grid_predict))
        print('Confusion Matrix : \n', mt.confusion_matrix(self.y_test, grid_predict))
        print('F1 Score : ', mt.f1_score(self.y_test, grid_predict,
                                         average='weighted', labels=np.unique(grid_predict)))
        print('Precision : ', mt.precision_score(self.y_test, grid_predict, average='weighted',
                                                 labels=np.unique(grid_predict)))
