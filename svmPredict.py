import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import database as db
import gridSearch as gS


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
        # print(self.df.head())


        # Feature selection
        self.array = self.df.values
        # seperate feature and label
        self.X = self.array[:, 1:4]  #value X for feature
        self.y = self.array[:,4]   #value Y for label
        self.y = self.y.astype('int') #change value output to inetger data type
        self.model = LogisticRegression(solver='lbfgs', max_iter=10000)
        self.rfe = RFE(self.model, 2) #set limit optimal feature that can be used
        self.fit = self.rfe.fit(self.X, self.y) #find optimal parameter
        # print((self.fit.n_features_))
        # print((self.fit.support_))
        # print((self.fit.ranking_))

        #prediction using SVM-GridSearch
        predictGS = gS.GridSearch(self.X, self.y)
        predictGS.grid_func()











