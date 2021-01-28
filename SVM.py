import pandas as pd
import numpy as np
from sklearn import svm
import database as db


class Prediction:
    # initiation class Databaase from database.py
    archive = db.Database()

    def prediction_SVM(self):
        # make dataframe from database climate.db
        self.df = pd.read_sql_query('select * from rain', self.archive.conn)

        #Transforming data
        self.df = self.df.replace(', ' , '-', regex= True)
        print(self.df.head())
        print(self.df.dtypes)










