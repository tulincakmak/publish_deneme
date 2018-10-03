# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:00:19 2018

@author: tulincakmak
"""



import Levenshtein
import pandas as pd
import numpy as np



cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
cursor = cnxn.cursor()

sql="select * from [otelz__1521458750].[dbo].[pred_data_click]"
controlled = pd.read_sql(sql,cnxn)
will_be_checked = pd.read_sql(sql,cnxn)


data=pd.DataFrame(columns=['Id','Similarity']) 


for i in will_be_checked['id']:
    df=will_be_checked.loc[will_be_checked['id']==i]
    str1=df['RoomSpec'].to_string(index=False)
    for i2 in controlled['RoomSpec']:
        d=pd.DataFrame(data=[[i ,Levenshtein.distance(str1, i2)]], columns=['Id', 'Similarity'])
        data=data.append(d)
      
     
Levenshtein.distance('Tül', 'Tülin')       

from sqlalchemy import create_engine
engine = create_engine('mssql+pymssql://gizemaras:gzmrs@123@78.40.231.196/TempData', echo=False) 
data.to_sql('levensthein', con=engine, if_exists='append',chunksize=1000)
engine.execute("SELECT * FROM levensthein").fetchall()



conn = 'mssql+pymssql://gizemaras:gzmrs@123@78.40.231.196/TempData'
engine = sqlalchemy.create_engine(conn, isolation_level="READ COMMITTED", echo=False, pool_size=50, max_overflow=0)
conn = engine.connect()
data = pd.read_sql(sql, conn)


