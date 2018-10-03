# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:11:50 2018

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:07:36 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import datetime as dt
from datetime import timedelta
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data=pd.read_excel('sales_everthing_18052018.xlsx')


print(data['booking_value_index'].unique())
mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)


print(data['Bölge'].unique())
most_repeated_bolge=data['Bölge'].value_counts()
data['Bölge']=data['Bölge'].fillna(most_repeated_bolge.index[0])


def miss(x):
    return sum(x.isnull())

missings=data.apply(miss, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)

data=pd.get_dummies (data, columns= ['Bölge'], drop_first=False) 
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)
#data=pd.get_dummies(data, columns=['booking_value_index'], drop_first=False)
data=pd.get_dummies(data, columns=['HotelTypes'], drop_first=False)


test=data.loc[data['log_date']=='2018-05-16']
train=data.loc[data['log_date']<'2018-05-16']

test = test.sample(frac=1).reset_index(drop=True)

train = train.sample(frac=1).reset_index(drop=True)


will_be_dropped={'log_date', 'trivagoID'}
train2=train.drop(will_be_dropped, axis=1)

will_be_dropped={'log_date', 'trivagoID', 'TargetNetTotalCost'}
test2=test.drop(will_be_dropped, axis=1)

y=train2['TargetNetTotalCost']
X=train2.drop('TargetNetTotalCost', axis=1)

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


y_pred2=xgb.predict(test2)
mse_score=mse(test['TargetNetTotalCost'],y_pred2)
print(mse_score)
r2_score=r2(test['TargetNetTotalCost'],y_pred2)
print(r2_score)
mae_err=mae(test['TargetNetTotalCost'],y_pred2)   
print(mae_err)   


y_pred2=pd.DataFrame(data=y_pred2)
result=pd.concat([y_pred2,test['trivagoID'],test['TargetNetTotalCost']], axis=1)
result.to_excel('result_17052018.xlsx', index=False)


control=pd.read_excel('everything_test_result.xlsx')

r2_score=r2(control['clicks_actual'],control['clicks_pred'])
print(r2_score)


#cross validation
scores = cross_val_score(xgb, X_train, y_train, cv=10)
 print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
