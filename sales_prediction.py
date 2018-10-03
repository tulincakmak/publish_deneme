# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:53:48 2018

@author: tulincakmak
"""

import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import numpy as np


data=pd.read_excel('sales_21052018_2.xlsx')


data['content_score']=data['content_score'].fillna(data['content_score'].mean())
data['survey_score']=data['survey_score'].fillna(data['survey_score'].mean())
data['total_points']=data['total_points'].fillna(data['total_points'].mean())

most_repeated_bolge=data['bolge'].value_counts()
data['bolge']=data['bolge'].fillna(most_repeated_bolge.index[0])
data['stars']=data['stars'].fillna(data['stars'].mean())

data['std7net_total']=data['std7net_total'].fillna(data['std7net_total'].mean())
data['std15net_total']=data['std15net_total'].fillna(data['std15net_total'].mean())
data['std30net_total']=data['std30net_total'].fillna(data['std30net_total'].mean())
data['std45net_total']=data['std45net_total'].fillna(data['std45net_total'].mean())

def miss(x):
    return sum(x.isnull())

missings=data.apply(miss, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)

data=pd.get_dummies (data, columns= ['weekday'], drop_first=False) 
#data=pd.get_dummies (data, columns= ['MONTH'], drop_first=False) 
data=pd.get_dummies (data, columns= ['bolge'], drop_first=False) 

mapping={'1': 'kis','2': 'kis','3': 'bahar','4': 'bahar','5': 'bahar','6': 'yaz','7': 'yaz','8': 'yaz',
         '9': 'sonbahar','10': 'sonbahar','11': 'sonbahar','12': 'kis'}
data['MONTH']=data['MONTH'].map(mapping)

data=pd.get_dummies(data, columns=['MONTH'], drop_first=False)

data=data.drop('created_on', axis=1)
data = data.sample(frac=1).reset_index(drop=True)
data=data.drop('trivago_id', axis=1)

X=data.drop('net_total', axis=1)
y=data['net_total']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

from xgboost import XGBRegressor 
xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=X_train, y=y_train, eval_metric=['rmse'])


y_pred=xgb.predict(X_val)
mse_score=mse(y_val,y_pred)
print(mse_score)
r2_score=r2(y_val,y_pred)
print(r2_score)
mae_err=mae(y_val,y_pred)   
print(mae_err)  

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(xgb.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances)

from sklearn.model_selection import GridSearchCV
n_estimators = [50, 100, 150, 200]
max_depth = [2, 3, 4, 6, 7, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
grid_search = GridSearchCV(xgb, param_grid, scoring="r2", n_jobs=-1,  verbose=1)
grid_result = grid_search.fit(X_train, y_train)
grid_result.best_params_
grid_result.best_score_