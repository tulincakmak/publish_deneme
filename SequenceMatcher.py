# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:13:54 2018

@author: tulincakmak
"""

from difflib import SequenceMatcher


import pandas as pd
import numpy as np



cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
cursor = cnxn.cursor()

sql="select * from [otelz__1521458750].[dbo].[pred_data_click]"
controlled = pd.read_sql(sql,cnxn)
will_be_checked = pd.read_sql(sql,cnxn)


data=pd.DataFrame(columns=['Id','Text', 'Similarity']) 


for i in will_be_checked['id']:
    df=will_be_checked.loc[will_be_checked['id']==i]
    str1=df['RoomSpec'].to_string(index=False)
    for i2 in controlled['RoomSpec']:
        u=SequenceMatcher(None, str1, i2)
        d=pd.DataFrame(data=[[i ,i2, u.ratio()]], columns=['Id', 'Text', 'Similarity'])
        data=data.append(d)
        
from sqlalchemy import create_engine
engine = create_engine('mssql+pymssql://gizemaras:gzmrs@123@78.40.231.196/TempData', echo=False) 
data.to_sql('levensthein_SequenceMatcher', con=engine, if_exists='append',chunksize=1000)
engine.execute("SELECT * FROM levensthein").fetchall()        