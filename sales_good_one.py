# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:02:33 2018

@author: tulincakmak
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
import pyodbc
from scipy import stats
import h2o


cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
cursor = cnxn.cursor()

sql="select * from ##newdata2"
data = pd.read_sql(sql,cnxn)

data=data.drop(['facility_id', 'cost', 'gross_rev', 'top_pos_share', 'hotel_impr','maxdate' ], axis=1)
data=data.drop('otelId', axis=1)


data['net_total_cost'] = data['net_total_cost'].fillna(0)

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


filter_col = [col for col in data if col.endswith('max')]
for i in range(len(filter_col)):
    data[filter_col[i]] = data[filter_col[i]].fillna(0)
    
object_cols=data.dtypes
convert_float=[]

for i in object_cols.index:
    if object_cols[i]== 'object' and i!='log_date' and i!='hotel_types' and i!='region' and i!='bolge':
        convert_float.append(i)
        
        
for i in convert_float:
    data[i]=pd.to_numeric(data[i], errors='ignore')    
    
    
data["cumsum_net_total_cost_max"]=data.groupby(["trivago_id"])["net_total_cost_max"].cumsum()
data["cumsum_total_night_max"]=data.groupby(["trivago_id"])["total_night_max"].cumsum()
data["cumsum_total_rooms_max"]=data.groupby(["trivago_id"])["total_rooms_max"].cumsum()
data["cumsum_myminprice_max"]=data.groupby(["trivago_id"])["myminprice_max"].cumsum()
data["cumsum_total_minprice_max"]=data.groupby(["trivago_id"])["total_minprice_max"].cumsum()
data["cumsum_clicks_max"]=data.groupby(["trivago_id"])["clicks_max"].cumsum()
data["cumsum_hotel_impr_max"]=data.groupby(["trivago_id"])["hotel_impr_max"].cumsum()
data["cumsum_gross_rev_max"]=data.groupby(["trivago_id"])["gross_rev_max"].cumsum()
data["cumsum_cnt_1_max"]=data.groupby(["trivago_id"])["cnt_1_max"].cumsum()
data["cumsum_cnt_2_max"]=data.groupby(["trivago_id"])["cnt_2_max"].cumsum()
data["cumsum_cnt_3_max"]=data.groupby(["trivago_id"])["cnt_3_max"].cumsum()
data["cumsum_cnt_4_max"]=data.groupby(["trivago_id"])["cnt_4_max"].cumsum()
data["cumsum_bid_max"]=data.groupby(["trivago_id"])["bid_max"].cumsum()    

filter_col = [col for col in data if col.startswith('stdev')]
for i in range(len(filter_col)):
    data[filter_col[i]] = data[filter_col[i]].fillna(data[filter_col[i]].mean())

data['hotel_types'] = data['hotel_types'].fillna(0)
data['hotel_types'].unique()
data['hotel_types']=data['hotel_types'].fillna('Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer Hotels', 'Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('Summer ', 'Summer Hotel')
data['hotel_types']=data['hotel_types'].replace('City Hotels', 'City Hotel')
data=pd.get_dummies(data, columns=['hotel_types'], drop_first=False)



for i in data.columns:    
    data[i] = data[i].fillna(0)
 
sil = ['clicks','avg_booking_value', 'cnt_1','cnt_2','cnt_3','cnt_4','cnt_total', 'diff', 'total_night',
'total_rooms','person', 'person', 'bid','myminposition', 'myminprice','top4minprice','total_minprice']

data = data.drop(sil, axis = 1)
data = data.drop('region', axis = 1)

sales = data

#stdeleri normalize et

filter_normalize = [col for col in sales if col.startswith('stdev')]
for i in sales[filter_normalize].columns:
    sales[i] = sales[i]/max(sales[i])

filter_normalize_2 = [col for col in sales if col.startswith('avg')]
for i in sales[filter_normalize].columns:
    sales[i] = sales[i]/max(sales[i])

bolge=sales['bolge'].value_counts()
sales['bolge']=sales['bolge'].fillna(bolge[0])

sales=pd.get_dummies(sales, columns=['bolge'], drop_first=False)

sales.info()

act_test=sales.loc[sales['log_date']=='2018-07-15']
train=sales.loc[sales['log_date']<'2018-07-15']

train2 = train.drop(['trivago_id','log_date'], axis = 1)
act_test2 = act_test.drop(['trivago_id','log_date'], axis = 1)


#y_train= train2['net_total_cost']
#train1= train2.drop('net_total_cost', axis = 1)

#y_test= test2['net_total_cost']
#test1= test2.drop('net_total_cost', axis = 1)


h2o.init()

hf = h2o.H2OFrame(train2)
tst = h2o.H2OFrame(act_test2)
train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)


y = 'net_total_cost'  
x = train.col_names
x.remove(y)



#GBR
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator(model_id="gbm_covType_v1",
    seed=2000000
    )
gbm.train(x=x, y=y, training_frame=train, validation_frame=valid)

#gbm.cross_validation_models()
#gbm.cross_validation_metrics_summary()
gbm.varimp_plot()
features=gbm.varimp()
features2=pd.DataFrame(list(features), columns=['cols','relative_importance','scaled_importance	','percentage'])

df2=pd.DataFrame()
for i in features2['percentage'].unique():
    if i<0.0051:
        df2=df2.append(features2.loc[features2['percentage']==i])

will_be_dropped=df2['cols']
train2=train2.drop(['bolge_Güneydoğu Anadolu' ,'bolge_Doğu Anadolu' ,'bolge_İç Anadolu'], axis=1)  

for  i in will_be_dropped:
    if  i!='bolge_Güneydoğu Anadolu' and i!='bolge_Doğu Anadolu' and i!='bolge_İç Anadolu':  
        train2=train2.drop(i, axis=1)

        

gbm.mse(train=True, valid=True, xval=False)
gbm.r2(train=True, valid=True, xval=False)
gbm.mae(train=True, valid=True, xval=True)

gbm.score_history()


y_act=act_test['net_total_cost']
y_act=y_act.reset_index()
y_act=y_act.drop(['index'], axis=1)

act_test['net_total_cost'].sum()

#RANDOMFOREST
from h2o.estimators.random_forest import H2ORandomForestEstimator
drf = H2ORandomForestEstimator()
drf.train(x=x, y = y, training_frame=train, validation_frame=valid)

drf.varimp_plot()
drf.varimp()

drf.mse(train=True, valid=True, xval=True)
drf.r2(train=True, valid=True, xval=True)
drf.mae(train=True, valid=True, xval=True)



calc=pd.read_csv('b12ae50697f1.csv')
r2_test=r2(act_test['net_total_cost'],calc)
print(r2_test)

h2o.shutdown(prompt=False)