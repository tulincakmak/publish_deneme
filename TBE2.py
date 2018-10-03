# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:51:56 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor 
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

train_data=pd.read_excel('tbe_train.xlsx')
test_data=pd.read_excel('tbe_test.xlsx')

train_data.head()

def missing(x):
    return sum(x.isnull())
    
print (train_data.apply(missing, axis=0))
print (test_data.apply(missing, axis=0))

train_data['outbid_ratio']=train_data['outbid_ratio'].fillna(0)
test_data['outbid_ratio']=test_data['outbid_ratio'].fillna(0)

mapping={'Low':1, 'Below Average':2, 'Avarage':3, 'Above Avarage':4, 'High':5}
train_data['booking_value_index']=train_data['booking_value_index'].map(mapping)
test_data['booking_value_index']=test_data['booking_value_index'].map(mapping)

train_data=train_data.drop('region', axis=1)
test_data=test_data.drop('region', axis=1)

train_data=train_data.drop('profit', axis=1)
test_data=test_data.drop('profit', axis=1)


train_data=train_data.drop('avg_cpc', axis=1)
test_data=test_data.drop('avg_cpc', axis=1)

train_data=train_data.drop('cost', axis=1)
test_data=test_data.drop('cost', axis=1)

x = test_data['clicks']

if stats.skew(x)>1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)')
   
elif stats.skew(x) <-1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')

elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')

elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')


train_data=train_data+1
for i in train_data.columns:
    train_data[i],_=stats.boxcox(train_data[i])  
    
test_data=test_data+1
for i in train_data.columns:
    test_data[i],_=stats.boxcox(test_data[i])    
   
train_data=np.log(train_data+1) 
test_data=np.log(test_data+1)   
    
y_train=train_data['clicks']
train_data=train_data.drop('clicks', axis=1)

y_test=test_data['clicks']
test_data=test_data.drop('clicks', axis=1)



xgb=XGBRegressor(learning_rate=0.1, max_depth =7)
xgb.fit(X=train_data, y=y_train)
y_pred=xgb.predict(test_data)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)
