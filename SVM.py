import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import database as db


class Prediction:

    def prediction_SVM(self):
        # make dataframe from database climate.db
        self.df = pd.read_csv('D:\Prediction_Thesis\csv\Climate.csv')

        #Transforming data
        self.df = self.df.replace(', ' , '-', regex= True)
        self.df['planting_time'] = LabelEncoder.fit_transform(self.df['planting_time'])
        self.df_enc = pd.get_dummies(self.df, columns=["region"])
        pd.set_option('display.max_columns', None)
        print(self.df_enc.head())
        print(self.df_enc.dtypes)










