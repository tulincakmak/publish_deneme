# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:51:10 2018

@author: tulincakmak
"""

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
    
columns_to_numeric=['avg7net_total_cost', 'avg15net_total_cost', 'avg30net_total_cost', 'avg45net_total_cost', 
                'avgnet_total_cost','avg15opp_cpc','avg30opp_cpc', 'avg45opp_cpc','bid','avg7opp_cpc','avgopp_cpc']

for i in columns_to_numeric:
    data[i]=pd.to_numeric(data[i], errors='ignore')
    
    
#correle_columns=['avg7cost', 'avg15cost', 'avg30cost', 'avg45cost', 'stdev30cost', 'stdev45cost', 'avgcost', 'stdevcost', 
#                 'avg7bookings', 'avg15bookings', 'avg30bookings', 'avg45bookings', 'stdev30bookings', 'stdev45bookings', 
#                 'avgbookings', 'stdevbookings', 'avg7profit', 'avg15profit', 'avg30profit', 'avg45profit', 'stdev30profit', 
#                 'stdev45profit', 'avgprofit', 'stdevprofit', 'avg7booking_value', 'avg15booking_value', 'avg30booking_value', 
#                 'avg45booking_value', 'stdev30booking_value', 'stdev45booking_value', 'avgbooking_value', 'stdevbooking_value',
#                 'avg7gross_rev', 'avg15gross_rev', 'avg30gross_rev', 'avg45gross_rev', 'stdev30gross_rev', 'stdev45gross_rev',
#                 'avggross_rev', 'stdevgross_rev']    
#
#data=data.drop(correle_columns, axis=1)

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
    if perc['percent'][k]>0.5 and i!='net_total_cost':
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
data['clicks']=data['clicks'].fillna(0)
data['score']=data['score'].fillna(data['score'].mean())

print(data.apply(get_missing))


data=data.drop ('facility_id', axis=1)


test=data.loc[data['created_on']=='2018-06-27']
train=data.loc[data['created_on']<'2018-06-27']

train1=train.loc[train['class']==1]
train1=train1.reset_index(drop=True)
train2=train.loc[train['class']==2]
train2=train2.reset_index(drop=True)
train0=train.loc[train['class']==0]
train0=train0.reset_index(drop=True)


X1=train1.drop(['trivago_id', 'class', 'net_total_cost', 'created_on'], axis=1)
X2=train2.drop(['trivago_id', 'class',  'net_total_cost', 'created_on'], axis=1)
X0=train0.drop(['trivago_id', 'class',  'net_total_cost', 'created_on'], axis=1)


y1=train1['net_total_cost']
y2=train2['net_total_cost']
y0=train0['net_total_cost']


from sklearn.model_selection import train_test_split
X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.25)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.25)
X0_train, X0_val, y0_train, y0_val = train_test_split(X0, y0, test_size=0.25)


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

from xgboost import XGBRegressor 
xgb0=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb1=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb2=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)


xgb1.fit(X=X1_train, y=y1_train, eval_metric=['rmse'])
y_pred1=xgb1.predict(X1_val)
mse_score=mse(y1_val,y_pred1)
print(mse_score)
r2_score=r2(y1_val,y_pred1)
print(r2_score)
mae_err=mae(y1_val,y_pred1)   
print(mae_err) 


xgb2.fit(X=X2_train, y=y2_train, eval_metric=['rmse'])
y_pred2=xgb2.predict(X2_val)
mse_score=mse(y2_val,y_pred2)
print(mse_score)
r2_score=r2(y2_val,y_pred2)
print(r2_score)
mae_err=mae(y2_val,y_pred2)   
print(mae_err)  

xgb0.fit(X=X0_train, y=y0_train, eval_metric=['rmse'])
y_pred0=xgb0.predict(X0_val)
mse_score=mse(y0_val,y_pred0)
print(mse_score)
r2_score=r2(y0_val,y_pred0)
print(r2_score)
mae_err=mae(y0_val,y_pred0)   
print(mae_err) 

importances = pd.DataFrame({'feature':X0_train.columns,'importance':np.round(xgb0.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).reset_index()
#importances.to_excel('sales_pred_feature_importance.xlsx')
print(importances.to_excel('importance_sales_2806_3.xlsx'))



test1=test.loc[test['class']==1]
test1=test1.reset_index(drop=True)
test2=test.loc[test['class']==2]
test2=test2.reset_index(drop=True)
test0=test.loc[test['class']==0]
test0=test0.reset_index(drop=True)

test11=test1.drop(['trivago_id', 'class', 'net_total_cost', 'created_on'], axis=1)
test22=test2.drop(['trivago_id', 'class', 'net_total_cost', 'created_on'], axis=1)
test00=test0.drop(['trivago_id', 'class', 'net_total_cost', 'created_on'], axis=1)

y_pred0=xgb0.predict(test00)
y_pred1=xgb1.predict(test11)
y_pred2=xgb2.predict(test22)

y_pred0=pd.DataFrame(data=y_pred0)
y_pred1=pd.DataFrame(data=y_pred1)
y_pred2=pd.DataFrame(data=y_pred2)

result0=pd.concat([y_pred0, test0[['bid', 'trivago_id']]], axis=1, join='inner')
result1=pd.concat([y_pred1, test1[['bid', 'trivago_id']]], axis=1, join='inner')
result2=pd.concat([y_pred2, test2[['bid', 'trivago_id']]], axis=1, join='inner')

df_pred=pd.concat([result0, result1, result2], axis=0)
df_pred=df_pred.reset_index(drop=True)
df_actual=pd.concat([test0[['bid', 'trivago_id']], test1[['bid', 'trivago_id']], test2[['bid', 'trivago_id']]])
df_actual=df_actual.reset_index(drop=True)

df=pd.concat([df_pred, df_actual], axis=1)

df_pred.to_excel('sales_pred_result_2706.xlsx')