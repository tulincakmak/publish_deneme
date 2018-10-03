# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:07:29 2018

@author: tulincakmak
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:08:30 2018

@author: tulincakmak
"""

import pandas as pd
import pymrmr
from xgboost import XGBRegressor 
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler
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


data=pd.read_excel('ForGulsahMergedData.xlsx')


def missing(x):
    return sum(x.isnull())

print (data.apply(missing, axis=0))



mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)


data=pd.get_dummies (data, columns= ['month'], drop_first=False) 

data=data.drop('trivagoId', axis=1)


test_data=data.loc[data['log_date']=='2018-04-09']
train_data=data.loc[data['log_date']!='2018-04-09']

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

columns=pymrmr.mRMR(data,'MIQ',10)

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


y_train=train_data['clicks']
train_data=train_data.drop('clicks', axis=1)

y_test=test_data['clicks']
test_data=test_data.drop('clicks', axis=1)

    
#train_data=train_data+1
#for i in train_data.columns:
#    train_data[i],_=stats.boxcox(train_data[i])  

#test_data=test_data+1
#for i in train_data.columns:
#    test_data[i],_=stats.boxcox(test_data[i])       


train_data=np.log(train_data+1) 
test_data=np.log(test_data+1) 

y=data['clicks']
X=data.drop('clicks', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    


xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=X_train, y=y_train)
y_pred=xgb.predict(X_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)

feat_selector = BorutaPy(rf, n_estimators=1000, verbose=2, random_state=1)


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
    models(i, X_train, y_train, X_test, y_test)
 

#cross validation 
seed=7
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(xgb,X_train, y_train, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

print(results.mean())

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