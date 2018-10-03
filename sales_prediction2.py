# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:50:17 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np

data=pd.read_excel('sales_everything_closest.xlsx')

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

will_be_dropped=[]
k=0
for i in perc['columns']:
    if perc['percent'][k]>0.5:
        will_be_dropped.append(i)
    k=k+1    
    

data=data.drop(will_be_dropped , axis=1)
  
columns= data.columns

avg_columns=[]
for  i in columns:
    if i.startswith('avg'):
        avg_columns.append(i)

std_columns=[]
for  i in columns:
    if i.startswith('std'):
        std_columns.append(i)


for i in avg_columns:
    data[i]=data[i].fillna(0)  
    
for k in std_columns:
    data[k]= data[k].fillna(data[k].mean())
    
data=pd.get_dummies(data, columns=['weekday'], drop_first=False)



data=data.drop ('facility_id', axis=1)
data=data.drop('closest_1', axis=1)
data=data.drop('closest_2', axis=1)
data=data.drop('closest_3', axis=1)



test2=data.loc[data['created_on']=='2018-05-24']
test2=test2.reset_index(drop=True)
test3=data.loc[data['created_on']=='2018-05-25']
test3=test3.reset_index(drop=True)
test4=data.loc[data['created_on']=='2018-05-26']
test4=test4.reset_index(drop=True)
test5=data.loc[data['created_on']=='2018-05-27']
test5=test5.reset_index(drop=True)
test6=data.loc[data['created_on']=='2018-05-28']
test6=test6.reset_index(drop=True)
test7=data.loc[data['created_on']=='2018-05-29']
test7=test7.reset_index(drop=True)


test8=data.loc[data['created_on']=='2018-06-03']
test8=test8.reset_index(drop=True)

test=test8.drop(['created_on', 'trivago_id', 'net_total_cost'], axis=1)
test=test.reset_index(drop=True)

train=data.loc[data['created_on']<'2018-06-03']


test22=test2.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test33=test3.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test44=test4.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test55=test5.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test66=test6.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test77=test7.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)
test88=test8.drop(['created_on','net_total_cost', 'trivago_id'], axis=1)


train2=train.drop(['created_on', 'trivago_id'], axis=1)

columns_to_numeric=['avg7net_total_cost', 'avg15net_total_cost', 'avg30net_total_cost', 'avg45net_total_cost', 
                    'avgnet_total_cost', 'closest1_net_total7', 'closest1_net_total30', 'closest1_net_total45', 
                    'closest2_net_total7', 'closest2_net_total30', 'closest2_net_total45', 'closest3_net_total7', 
                    'closest3_net_total30', 'closest3_net_total45']

for i in columns_to_numeric:
    train2[i]=pd.to_numeric(train2[i], errors='ignore')
    
for i in columns_to_numeric:
    test[i]=pd.to_numeric(test[i], errors='ignore')



X=train2.drop('net_total_cost', axis=1)
y=train2['net_total_cost']

#import scipy.stats as stats
#
#x = data['net_total_cost']
#if stats.skew(x)>1:
#   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)') 
#elif stats.skew(x) <-1:
#   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')
#elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
#   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')
#elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
#   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')
#   
#y2=np.log(y+1)    


X.info()

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

from xgboost import XGBRegressor 
xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=X_train, y=y_train, eval_metric=['rmse'])


y_pred2=xgb.predict(X_test)
mse_score=mse(y_test,y_pred2)
print(mse_score)
r2_score=r2(y_test,y_pred2)
print(r2_score)
mae_err=mae(y_test,y_pred2)   
print(mae_err)  

y_pred=pd.DataFrame(data=y_pred)
y_val=y_val.reset_index(drop=True)
y_pred=y_pred.reset_index(drop=True)

r=pd.concat([y_pred,y_val], axis=1)
r.to_excel('r.xlsx')

y_pred2=xgb.predict(test22)
y_pred3=xgb.predict(test33)
y_pred4=xgb.predict(test44)
y_pred5=xgb.predict(test55)
y_pred6=xgb.predict(test66)
y_pred7=xgb.predict(test77)
y_pred8=xgb.predict(test)

y_pred2=pd.DataFrame(data=y_pred2)
y_pred3=pd.DataFrame(data=y_pred3)
y_pred4=pd.DataFrame(data=y_pred4)
y_pred5=pd.DataFrame(data=y_pred5)
y_pred6=pd.DataFrame(data=y_pred6)
y_pred7=pd.DataFrame(data=y_pred7)
y_pred8=pd.DataFrame(data=y_pred8)

y_pred2=np.exp(y_pred2)
y_pred3=np.exp(y_pred3)
y_pred4=np.exp(y_pred4)
y_pred5=np.exp(y_pred5)
y_pred6=np.exp(y_pred6)
y_pred7=np.exp(y_pred7)
y_pred8=np.exp(y_pred8)


result2=pd.concat([y_pred2, test2[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result3=pd.concat([y_pred3, test3[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result4=pd.concat([y_pred4, test4[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result5=pd.concat([y_pred5, test5[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result6=pd.concat([y_pred6, test6[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result7=pd.concat([y_pred7, test7[['net_total_cost', 'trivago_id']]], axis=1, join='inner')
result8=pd.concat([y_pred8, test8[['net_total_cost', 'trivago_id']]], axis=1, join='inner')

result2.to_excel('result24_log.xlsx')
result3.to_excel('result25_log.xlsx')
result4.to_excel('result26_log.xlsx')
result5.to_excel('result27_log.xlsx')
result6.to_excel('result28_log.xlsx')
result7.to_excel('result29_log.xlsx')
result8.to_excel('result04062018_log.xlsx')




importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(xgb.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).reset_index()
importances.to_excel('sales_pred_feature_importance.xlsx')



#importance ı 0 olan kolonlar droplanıyor.
will_be_dropped=['avg45cnt_2',
'closest3_total_room45',
'stdev45total_votes',
'stdev45total_points',
'avg15cnt_4',
'stdev45survey_score',
'stdev30total_votes',
'avg45cnt_3',
'stdev30total_points',
'avg30cnt_3',
'avg30cnt_4',
'stdev30survey_score',
'stdev30score',
'avg45cnt_4',
'stdev45score']

train=train.drop(will_be_dropped, axis=1)



X=train.drop('net_total_cost', axis=1)
y=train['net_total_cost']


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

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

y_pred=pd.DataFrame(data=y_pred)
result=pd.concat([y_pred, y_val], axis=1, join='inner')