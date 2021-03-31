import pandas as pd
import numpy as np
import collections as col
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import database as db
import gridSearch as gS
import nelderMead as nM
from imblearn.over_sampling import SMOTE


class Prediction:

    def predict(self):
        # Load dataframe from csv
        self.df = pd.read_csv('D:\Prediction_Thesis\csv\Climate.csv')

        # Transforming data
        self.df = self.df.replace(', ' , '-', regex= True)  #replace all comma to dash in dataframe
        self.df['date'] = pd.to_datetime(self.df['date'])
        le = LabelEncoder() #initiation LabelEncoder
        oe = OrdinalEncoder() #initiation OrdnialEncoder
        self.temp = np.array(self.df['region']).reshape(-1,1) #reshape 'region' value
        self.df['planting_time'] = le.fit_transform(self.df['planting_time'])  #change value in 'planting_time'
        self.df['region'] = oe.fit_transform(self.temp) #change value in 'region'
        # self.df = self.df.drop('date')
        self.X = self.df.iloc[:, 2:4].values
        self.y = self.df.iloc[:, 4].values
        # Resample minority class data
        smote = SMOTE()
        X_sm, y_sm = smote.fit_resample(self.X, self.y)
        self.X = X_sm
        self.y = y_sm
        total_data = len(self.y)
        # print(self.df.head())


        #prediction using SVM-GridSearch
        predictGS = gS.Optimization(X_sm, y_sm)
        predictGS.grid_func()



        predictNM =nM.Optimization(X_sm, y_sm, total_data)
        predictNM.nm_func()











