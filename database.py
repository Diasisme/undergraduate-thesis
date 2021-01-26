import sqlite3


class Database:
    def __init__(self):
        database_path = "D:/Prediction_Thesis/database/climate.db"
        self.conn = sqlite3.connect(database_path)
        self.cur = self.conn.cursor()

    def execute(self, query):
        self.cur.execute(query)
        self.conn.commit()
        print(self.cur.fetchall())
        return self.cur
