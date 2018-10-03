# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:58:44 2018

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


data=pd.read_excel('networkdr_cltv.xlsx')

data.head()

#Create a new function:
def num_missing(x):
  return sum(x.isnull())

print ("Missing values per column:")
print (data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#This returns both mode and count. Remember that mode can be an array as there can be multiple values with high frequency
mode(data['uc']).mode[0]
mode(data['alti']).mode[0]
mode(data['oniki']).mode[0]
mode(data['mean_six']).mode[0]
mode(data['mean_twelve']).mode[0]
         
data['uc'].fillna(mode(data['uc']).mode[0], inplace=True)        
data['alti'].fillna(mode(data['alti']).mode[0], inplace=True)        
data['oniki'].fillna(mode(data['oniki']).mode[0], inplace=True)  
data['mean_six'].fillna(mode(data['mean_six']).mode[0], inplace=True)        
data['mean_twelve'].fillna(mode(data['mean_twelve']).mode[0], inplace=True)    

data.info()
city_mapping = {'Istanbul':'Marmara','Kocaeli':'Marmara', 'Manisa':'Ege','Izmir':'Ege','Aydin':'Ege',
                'Denizli':'Ege','Antalya':'Akdeniz','Adana':'Akdeniz','Afyon':'Ege','Bursa':'Marmara','Mersin':'Akdeniz','Isparta':'Akdeniz',
                'Kayseri':'Icand','Eskisehir':'Icand', 'Kütahya':'Ege',
                'Konya':'Icand','Ankara':'Icand','Samsun':'Karadeniz','Gaziantep':'Doguand'}

data['il'] = data['il'].map(city_mapping)
data=pd.get_dummies (data, columns= ['Gender'], drop_first=True)
data=pd.get_dummies (data, columns= ['il'], drop_first=True)


data=data.drop('gsm',axis=1)


for k in data.columns:
    print( k, stats.pearsonr(data[k],data['uc']))

for k in data.columns:
    print( k, stats.spearmanr(data[k],data['uc']))


stats.jarque_bera(data['uc'])


#LOG TRANSFORMATION
#data=np.log(data+1)



#BOXCOX TRANSFORMATION
data=data+1
for i in data.columns:
    data[i],_=stats.boxcox(data[i])
    print(stats.skew(data[i]))




stats.kurtosis(data['uc'])
stats.skew(data['uc'])



data['uc'].plot(kind='hist',bins=100)
h = np.asarray(data['uc'].dropna())
h = sorted(h)
#use the scipy stats module to fit a normal distirbution with same mean and standard deviation
fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 
#plot both series on the histogram
plt.plot(h,fit,'-',linewidth = 2)
plt.hist(h,normed=True,bins = 100)      
plt.show() 

#z-score standardization
data = data.apply(zscore)

#MinMaxScaling best option for regression models

scaler=MinMaxScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_df, columns=data.columns)



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
ax1.set_title('Before Scaling')
for i in data.columns:
    sns.kdeplot(data[i], ax=ax1)

ax2.set_title('After Min-Max Scaling')
for i in data.columns:
    sns.kdeplot(scaled_df[i], ax=ax2)
plt.show()


#TRY XGBOOSTREGRESSION
y=data['uc']
X=data.drop(['uc'], axis=1)


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


xgb=XGBRegressor(learning_rate=0.1, max_depth =7)
xgb.fit(X=X_train, y=y_train)
y_pred=xgb.predict(X_test)
mse_score=mse(y_test,y_pred)
print(mse_score)
r2_score=r2(y_test,y_pred)
print(r2_score)


#XGBoost Hyperparameter Tuning
n_estimators = [50, 100, 150, 200,1000]
max_depth = [2, 3, 4, 6, 7, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(xgb, param_grid, scoring="r2", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))



#H2O
import h2o

h2o.init()

hf = h2o.H2OFrame(data)
train, valid, test = hf.split_frame([0.6, 0.2], seed=1234)


y = 'uc'  
x = train.col_names
x.remove(y)



from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator(nfolds=3)
gbm.train(x=x, y=y, training_frame=train, validation_frame=valid)

#gbm.cross_validation_models()
#gbm.cross_validation_metrics_summary()
gbm.varimp_plot()
gbm.varimp()

gbm.mse(train=True, valid=True, xval=True)
gbm.r2(train=True, valid=True, xval=True)

