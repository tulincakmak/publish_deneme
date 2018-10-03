# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:29:15 2018

@author: tulincakmak
"""


import pandas as pd
import numpy as np
from scipy.stats import mode
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import h2o
from xgboost import XGBRegressor 


data=pd.read_excel('networkdr_cltv.xlsx')

data['oniki'] = data['oniki'].fillna((data['oniki'].mean()))
data['uc'] = data['uc'].fillna((data['uc'].mean()))
data['alti'] = data['alti'].fillna((data['alti'].mean()))
data['mean_twelve'] = data['mean_twelve'].fillna((data['mean_twelve'].mean()))
data['mean_six'] = data['mean_six'].fillna((data['mean_six'].mean()))
data['il'] = data['il'].fillna('Istanbul')
 

city_mapping = {'Istanbul':'Marmara','Kocaeli':'Marmara', 'Manisa':'Ege','Izmir':'Ege','Aydin':'Ege',
                'Denizli':'Ege','Antalya':'Akdeniz','Adana':'Akdeniz','Afyon':'Ege','Bursa':'Marmara','Mersin':'Akdeniz','Isparta':'Akdeniz',
                'Kayseri':'Icand','Eskisehir':'Icand', 'Kütahya':'Ege',
                'Konya':'Icand','Ankara':'Icand','Samsun':'Karadeniz','Gaziantep':'Doguand'}

data['il'] = data['il'].map(city_mapping)
data=pd.get_dummies (data, columns= ['Gender'], drop_first=True)
data=pd.get_dummies (data, columns= ['il'], drop_first=True)


data=data.drop('gsm',axis=1)

#LOG TRANSFORMASYON
#


#BOXCOX TRANSFORMATION
data=data+1
for i in data.columns:
    data[i],_=stats.boxcox(data[i])
   
#MinMaxScaler    
scaler=MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
data=scaled_df


#z-score standardization
data = data.apply(zscore)


#for k in data.columns:
#    print( k, stats.pearsonr(data[k],data['uc']))
#
#for k in data.columns:
#    print( k, stats.spearmanr(data[k],data['uc']))

#calculate skewness
x = data['uc']

if stats.skew(x)>1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Right)')
   
elif stats.skew(x) <-1:
   print('Skewness is: '+ str(stats.skew(x))+'; Highly skewed(Left)')

elif (stats.skew(x)<0.5 and stats.skew(x)>-0.5):
   print('Skewness is: '+ str(stats.skew(x))+'; Symmetric')

elif (stats.skew(x)<1 and stats.stats.skew(x)>0.5) or (stats.skew(x)<-0.5 and stats.skew(x)>-1):
   print('Skewness is: '+ str(stats.skew(x))+'; Moderately skewed')



#jarque_bera = stats.jarque_bera(data['uc'])
#
#alpha = 0.05
#
#if (jarque_bera[1] < alpha):
#   print('p value is:'+str(jarque_bera[1])+' according to Jarque Bera, thus reject the null hyphotesis about normality.')
#else:
#   print('p value is:'+str(jarque_bera[1])+' according to Jarque Bera, thus do not reject the null hyphotesis about normality.')   
 
   
y=data['uc']
X=data.drop(['uc'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#SCİKİT LEARN
#GBR
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
mse_score=mse(y_test,gbr.predict(X_test))
print(mse_score)
r2_score=r2(y_test,gbr.predict(X_test))
print(r2_score)

#RANDOMFOREST
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
mse_score=mse(y_test,rf.predict(X_test))
print(mse_score)
r2_score=r2(y_test,rf.predict(X_test))
print(r2_score)


#H2O
h2o.init()

hf = h2o.H2OFrame(data)
train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)


y = 'uc'  
x = train.col_names
x.remove(y)


#GBR
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator()
gbm.train(x=x, y=y, training_frame=train, validation_frame=valid)

#gbm.cross_validation_models()
#gbm.cross_validation_metrics_summary()
gbm.varimp_plot()
gbm.varimp()

gbm.mse(train=True, valid=True, xval=False)
gbm.r2(train=True, valid=True, xval=False)


#RANDOMFOREST
from h2o.estimators.random_forest import H2ORandomForestEstimator
drf = H2ORandomForestEstimator()
drf.train(x=x, y = y, training_frame=train, validation_frame=valid)

drf.varimp_plot()
drf.varimp()

drf.mse(train=True, valid=True, xval=True)
drf.r2(train=True, valid=True, xval=True)


#XGBOOSTREGRESSOR
xgb=XGBRegressor(learning_rate=0.1, max_depth =7)
xgb.fit(X=X_train, y=y_train)
y_pred=xgb.predict(X_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)

