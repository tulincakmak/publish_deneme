# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:40:30 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

data=pd.read_excel('tbe_data.xlsx')

data.head()

drop_column=['otelName', 'reservationStart', 'reservationEnd','avgtopDistance', 'avgDistance'],

for i in drop_column:
    data=data.drop(i,axis=1)
    
data=pd.get_dummies (data, columns= ['Bölge'], drop_first=False) 


x = data['cliks']

if stats.skew(x)>1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)')
   
elif stats.skew(x) <-1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')

elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')

elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')


data=data+1
for i in data.columns:
    data[i],_=stats.boxcox(data[i])
    
scaler=MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
data=scaled_df    

y=data['cliks']
X=data.drop(['cliks'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
mse_score=mse(y_test,gbr.predict(X_test))
print(mse_score)
r2_score=r2(y_test,gbr.predict(X_test))
print(r2_score)


xgb=XGBRegressor(learning_rate=0.1, max_depth =7)
xgb.fit(X=X_train, y=y_train)
y_pred=xgb.predict(X_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)


rf = RandomForestRegressor()
rf.fit(X_train,y_train)
mse_score=mse(y_test,rf.predict(X_test))
print(mse_score)
r2_score=r2(y_test,rf.predict(X_test))
print(r2_score)
