# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:08:30 2018

@author: tulincakmak
"""


def hotel_based_split(df):
    ts = df.groupby('trivagoId').apply(lambda x : x.sample(frac=0.3, random_state=333))
    ts.index = ts.index.levels[1]
    tr = df.drop(ts.index)
    return tr, ts

import pandas as pd
import pymrmr
from xgboost import XGBRegressor 
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import KFold
import h2o
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor as ExtraTreeReg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

from boruta import BorutaPy

#select * into #temp FROM [TempData].[dbo].[trivago_otelz_report] where hotel_impr>0 and bid is not NULL
#
#select hotel_name,bid,status,opp_cpc,hotel_impr,clicks,avg_cpc,top_pos_share,outbid_ratio,beat,meet,lose,unavailability,max_potential,bookings,booking_rate,booking_value_index,replace(region,'Province','') as region,
#stars,rating,trivagoId,
# DATEDIFF (day,log_date,'2018-04-18') as holiday_diff, datepart(month,log_date) as month,DATENAME(DW,CONVERT(VARCHAR(20),log_date,101)) as weekday, 
# case when DATENAME(DW,CONVERT(VARCHAR(20),log_date,101))='Sunday' then 1 when DATENAME(DW,CONVERT(VARCHAR(20),log_date,101))='Saturday' then 1 else 0 END as isweekend,log_date
# --into tempdata.dbo.otelz_data_deneme3  
#from #temp
#
# --data preprocessing
# select bid,opp_cpc, hotel_impr,clicks, beat, meet, lose, unavailability, max_potential, booking_value_index, REPLACE(REPLACE(region,'()',''),'â','a') as region,stars, 
# rating, trivagoId,holiday_diff, month, weekday, isweekend,convert(date,log_date) as log_date
# into #tmp from tempdata.dbo.otelz_data_deneme3
#  
#select bid,opp_cpc, hotel_impr,clicks, beat, meet, lose, unavailability, max_potential, booking_value_index, region,stars, rating, trivagoId,holiday_diff, month, weekday, isweekend,log_date,Bölge
#into #temp2 from #tmp o left join  [CrawlerData].[dbo].[turkey_sections] t on tr.dbo.convertTurkish(o.region)=tr.dbo.convertTurkish(t.Ad)
#
#update   #temp2 set Bölge='Ege'  where region='Afyon'
#update   #temp2 set opp_cpc=bid where opp_cpc is NULL
#update  #temp2 set beat=1 where beat>1
#update  #temp2 set meet=1 where meet>1
#update  #temp2 set lose=1 where lose>1
#update #temp2 set rating=50 where rating is NULL

data=pd.read_excel('tbe_data.xlsx')

def missing(x):
    return sum(x.isnull())

print (data.apply(missing, axis=0))

data.groupby('Bölge')['Bölge'].count()

data['Bölge'].value_counts()

data.groupby('booking_value_index')['booking_value_index'].count()

data['Bölge']=data['Bölge'].fillna('Marmara')

mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

data['Bölge'] = data['Bölge'].str.strip()
data['weekday'] = data['weekday'].str.strip()

data=pd.get_dummies (data, columns= ['Bölge'], drop_first=False) 
data=pd.get_dummies (data, columns= ['weekday'], drop_first=False) 
data=pd.get_dummies (data, columns= ['month'], drop_first=False) 

data=data.drop('region', axis=1)
data=data.drop('trivagoId', axis=1)

test_data=data.loc[data['log_date']=='2018-04-03']
train_data=data.loc[data['log_date']!='2018-04-03']

test_data=test_data.drop('log_date', axis=1)
train_data=train_data.drop('log_date', axis=1)
data=data.drop('log_date', axis=1)

#
#h2o.init()
#hf = h2o.H2OFrame(train_data)
#train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)
#y = 'clicks'  
#x = train.col_names
#x.remove(y)
##GBR
#from h2o.estimators.gbm import H2OGradientBoostingEstimator
#gbm = H2OGradientBoostingEstimator()
#gbm.train(x=x, y=y, training_frame=train, validation_frame=valid)
#y_pred=gbm.predict(test_data)
##gbm.cross_validation_models()
##gbm.cross_validation_metrics_summary()
#gbm.varimp_plot()
#gbm.varimp()
#gbm.mse(train=True, valid=True, xval=False)
#gbm.r2(train=True, valid=True, xval=False)

result=pd.concat([y_train,train_data], axis=1)

columns=pymrmr.mRMR(result,'MIQ',10)

print(columns)

new_data=[]
new_data=pd.DataFrame(data=new_data)

new_data_test=[]
new_data_test=pd.DataFrame(data=new_data_test)

for i in columns:
    new_data_test=pd.concat([new_data_test,data[i]], axis=1)

for i in columns:
    new_data=pd.concat([new_data,data[i]], axis=1)
    

x = data['clicks']
if stats.skew(x)>1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)') 
elif stats.skew(x) <-1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')
elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')
elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')

    
#train_data=train_data+1
#for i in train_data.columns:
#    train_data[i],_=stats.boxcox(train_data[i])  

#test_data=test_data+1
#for i in train_data.columns:
#    test_data[i],_=stats.boxcox(test_data[i])       

train_data=np.log(train_data+1) 
test_data=np.log(test_data+1)  

    
y_train=train_data['clicks']
train_data=train_data.drop('clicks', axis=1)

y_test=test_data['clicks']
test_data=test_data.drop('clicks', axis=1)

x_train_matrix=train_data.as_matrix()
y_train_matrix=y_train.as_matrix()

rf = RandomForestRegressor(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(x_train_matrix,y_train_matrix)




xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=train_data, y=y_train)
y_pred=xgb.predict(new_data_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)

feat_selector = BorutaPy(xgb, n_estimators='balanced_subsample', verbose=2, random_state=1)


def models(model, x_train, y_train, x_test, y_test):
    estimator = model()
    estimator.fit(x_train,y_train)
    t=estimator.score(x_train,y_train)
    y_pred = estimator.predict(x_test)
    mse_score=mse(y_test,y_pred)
    print("mse_score: " + str(mse_score))
    r2_score=r2(y_test, y_pred)
    print("r2_score: " + str(r2_score))
    print(t)


mdl=[RandomForestRegressor, ExtraTreeReg, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor]

for i in mdl:
    models(i, new_data, y_train, new_data_test, y_test)
 

#cross validation 
seed=7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(xgb,train_data, y_train, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#hyperparameter optimization
n_estimators = [50, 100, 150, 200]
max_depth = [2, 3, 4, 6, 7, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(xgb, param_grid, scoring="r2", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(train_data, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


#graph
plt.plot(y_pred, 'r--', y_test, 'bs')
plt.show()