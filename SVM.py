import pandas as pd
import numpy as np
from sklearn import svm
import database as db


class Prediction:
    archive = db.Database()                                                                                 # initiation class from other file

    def prediction(self):
        self.df = pd.read_sql_query('select * from rain', self.archive.conn)                                # make dataframe from database climate.db
        self.df['plantingTime_value'] = np.where(self.df['planting_time']!= 'BURUK', True, False)           # make new output column
        self.df["plantingTime_value"].replace({"True":"0","False":"1"}, inplace= True)
        print(self.df.head())




predict = Prediction()

predict.prediction()





