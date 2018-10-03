# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:22:52 2018

@author: tulincakmak
"""
import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy import stats
import h2o
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae

test=pd.read_excel('everything_avarage2_testdata2.xlsx')

test['bölge'] = test['bölge'].str.strip()
test['booking_value_index'] = test['booking_value_index'].str.strip()

test.groupby('bölge')['bölge'].count()
test['bölge']=test['bölge'].fillna('Marmara')



test["bölge"]=test["bölge"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))


mapping2={'Marmara': 'Marmara', 'Ege': 'Ege', 'Akdeniz':'Akdeniz', 'Karadeniz':'Others', 'İçAnadolu':'Others', 'DoğuAnadolu': 'Others', 'GüneydoğuAnadolu': 'Others'}
test['bölge']=test['bölge'].astype(str).map(mapping2)

test=pd.get_dummies (test, columns= ['bölge'], drop_first=False)

test["booking_value_index"]=test["booking_value_index"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))

mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
test['booking_value_index']=test['booking_value_index'].map(mapping)

test=pd.get_dummies (test, columns= ['booking_value_index'], drop_first=False) 

test=test.drop('date_name', axis=1)


test['date_name_Tuesday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Wednesday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Thursday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Friday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Saturday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Sunday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)
test['date_name_Monday'] = pd.Series(np.random.randn(len(test['bid'])), index=test.index)

test['date_name_Tuesday']=0
test['date_name_Wednesday']=0
test['date_name_Thursday']=0
test['date_name_Friday']=0
test['date_name_Saturday']=0
test['date_name_Sunday']=0
test['date_name_Monday']=1

test['rating'] = test['rating'].fillna(test['rating'].mean)

test = test.sample(frac=1).reset_index(drop=True)

test2=test.drop('trivagoId', axis=1)



y_pred2=xgb.predict(test2)

y_pred2=pd.DataFrame(data=y_pred2.round())
result=pd.concat([test['trivagoId'], y_pred2], axis=1)
result.to_csv('prediction_results_2504.csv', index=False)

true_value=pd.read_excel('actual_value.xlsx')
pred=pd.read_excel('prediction_results_2504.xlsx')



mse_score=mse(true_value['clicks'],pred['clicks'])
print(mse_score)
r2_score=r2(true_value['clicks'],pred['clicks'])
print(r2_score)

pred.loc[pred['clicks']==-1]=0