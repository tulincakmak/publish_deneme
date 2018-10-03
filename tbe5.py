# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:07:35 2018

@author: tulincakmak
"""

import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
import numpy as np

data=pd.read_excel('CaglaExcel.xlsx')

bootstrapped_data=[]

def hotel_based_split(df, test_rate, seed):
  ts = df.groupby('trivagoId').apply(lambda x : x.sample(frac=test_rate, random_state=seed))
  ts.index = ts.index.levels[1]
  tr = df.drop(ts.index)
  return tr, ts



for seed in range(10):
  train_data, test_data = hotel_based_split(data, 0.25, seed)
  train_data, val_data = hotel_based_split(train_data, 0.1875, seed)
  bootstrapped_data.append( train_data)
  bootstrapped_data.append( val_data)
  bootstrapped_data.append(test_data)
  
  

def PrintLog(text):
     file = open("test.txt","a") 
     file.write(str(text)) 
     file.close()  
  
count=0

for boot in bootstrapped_data:
   if(count%3==0):
       X_tr=boot
       y_tr=X_tr['clicks']
       X_tr=X_tr.drop('clicks',axis=1)
       
   elif(count%3==1):
       
        X_va=boot
        y_va=X_va['clicks']
        X_va=X_va.drop('clicks',axis=1)
   
   elif(count%3==2):
       
       X_test=boot
       y_test=X_test['clicks']
       X_test=X_test.drop('clicks',axis=1)
       
   count=count+1
   CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
   CV_rfc.fit(X_tr, y_tr)
   rb=RandomForestRegressor(n_estimators=CV_rfc.best_estimators_.n_estimators,max_features=CV_rfc.best_estimators_.max_features,min_samples_split=CV_rfc.best_estimators_.min_samples_split, bootstrap=CV_rfc.best_estimators_.bootstrap  )
   rb.fit(X_tr,y_tr)
   y_pred = rb.predict(X_test)
   y_predVal = rb.predict(X_va)
   mse_score=mse(y_test,y_pred)
   mse_scoreVal=mse(y_va,y_predVal)
   rms = np.sqrt(mse_score)
   rmsVal = np.sqrt(mse_scoreVal)
   r2_score=r2(y_test, y_pred)
   r2_scoreVal=r2(y_va, y_predVal)
   PrintLog("For Dataset"+str(count-1) + "  r2_score: " + str(r2_score)+"   "+"root mean square error: "+str(rms)+"\n For Validation Dataset"+str(count-1) + "  r2_scoreVal: " + str(r2_scoreVal)+"   "+"root mean square errorVal: "+str(rmsVal)+"\n\n\n")