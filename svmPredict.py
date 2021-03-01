import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import database as db
import gridSearch as gS
import nelderMead as nM


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
        self.X = self.df.iloc[:, 2:4].values
        self.y = self.df.iloc[:, 4].values
        # print(self.df.head())


        #prediction using SVM-GridSearch
        predictGS = gS.Optimization(self.X, self.y)
        predictGS.grid_func()

        predictNM =nM.Optimization(self.X, self.y)
        predictNM.nm_func()











