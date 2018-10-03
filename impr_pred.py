# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:32:45 2018

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:51:30 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import scipy.stats as stats

data=pd.read_excel('everything_avarage_28052018.xlsx')

#null kontrolü
def get_missing(x):
    return(sum(x.isnull()))
    
missing=data.apply(get_missing) 

df=pd.DataFrame( columns=['columns', 'percent'])
i2=0
for i in missing:
    perc=(i/data['trivago_id'].count())
    df=df.append({'columns': missing.index[i2], 'percent':perc} , ignore_index=True)
    i2=i2+1

perc=df.sort_values(by=['percent'])  

perc=perc.reset_index()

will_be_dropped=[]
k=0
for i in perc['columns']:
    if perc['percent'][k]>0.5:
        will_be_dropped.append(i)
    k=k+1    
    

data=data.drop(will_be_dropped , axis=1)

bolge=data['bolge'].value_counts()
data['bolge']=data['bolge'].fillna(bolge[0])
data['stars']=data['stars'].fillna(3)
data['rating']=data['rating'].fillna(79)
data['hotel_types']=data['hotel_types'].fillna('Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer Hotels', 'Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer ', 'Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('City Hotels', 'City Hotel')
data['hotel_types'].unique()

#kategorik kolon kalmadığına emin olduktan sonra çalıştır.!!
missings=data.apply(get_missing, axis=0)    
fill_missing=missings[missings>0]
for i in fill_missing.index:
    data[i]=data[i].fillna(0)

 
columns= data.columns

avg_columns=[]
for  i in columns:
    if i.startswith('avg'):
        avg_columns.append(i) 
        
for i in avg_columns:
    data[i]=data[i].fillna(0)          
    
data=pd.get_dummies(data, columns=['bolge'], drop_first=False)
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)
data=pd.get_dummies(data, columns=['hotel_types'], drop_first=False)

print(data['booking_value_index'].unique())
data['booking_value_index']=data['booking_value_index'].replace('', 'Low')
mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

columns_to_numeric=['avg3click_hotelimpr', 'avg7click_hotelimpr', 'avg30click_hotelimpr']

for i in columns_to_numeric:
    data[i]=pd.to_numeric(data[i], errors='ignore')


data.info() 


object_cols=data.dtypes
convert_float=[]

for i in object_cols.index:
    if object_cols[i]== 'object' and i!='log_date':
        convert_float.append(i)
        
        
for i in convert_float:
    data[i]=pd.to_numeric(data[i], errors='ignore')

    
test=data.loc[data['log_date']=='2018-07-09']
train=data.loc[data['log_date']<'2018-07-09']

test_raw=test
train_raw=train

test_raw=test_raw.reset_index(drop=True)
train_raw=train_raw.reset_index(drop=True)

train=train.sample(frac=1).reset_index(drop=True)
test=test.sample(frac=1).reset_index(drop=True)


train2=train.drop(['log_date'], axis=1)
train2=train2.drop('trivago_id', axis=1)

train2=train2.drop(['avg_cpc','clicks'], axis=1)
test=test.drop(['avg_cpc','clicks'], axis=1)



X=train2.drop('hotel_impr', axis=1)
y=train2['hotel_impr']

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
importances = importances.sort_values('importance',ascending=False).reset_index()
print(importances)

df2=pd.DataFrame()
for i in importances['importance'].unique():
    if i<0.0051:
        df2=df2.append(importances.loc[importances['importance']==i])

will_be_dropped=df2['feature'] 
data2=data.drop(will_be_dropped, axis=1)       

importances.to_excel('click_feature_importance_4.xlsx') 
      
test=test.drop('log_date', axis=1)
test1=test.drop('trivago_id', axis=1)
test1=test1.drop('hotel_impr', axis=1)


y_pred1=xgb.predict(test1)    
y_pred1=pd.DataFrame(data=y_pred1)  

y_pred1=y_pred1.round()


result1=pd.concat([y_pred1, test_raw['trivago_id'], test_raw['hotel_impr'],test_raw['clicks']], axis=1)


result1.to_excel('impr_pred_deneme8_0907.xlsx')

#concat saçmalarsa burdan tek tek çıkart
y_test_pred=xgb.predict(X_test)
test_r2=r2(y_test, y_test_pred)
print(test_r2)

#x = data['clicks']
#if stats.skew(x)>1:
#   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)') 
#elif stats.skew(x) <-1:
#   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')
#elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
#   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')
#elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
#   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')  
#
##cross validation
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(xgb, X_train, y_train, cv=10)
