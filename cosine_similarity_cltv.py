# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:38:54 2018

@author: tulincakmak
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

data=pd.read_excel('cltv_cold_start.xlsx')

data.head()



def missing(x):
    return sum(x.isnull())
    
print (data.apply(missing, axis=0))

data.describe()

data['avgSalePrice'] = data['avgSalePrice'].fillna((data['avgSalePrice'].mean()))
data['avgRentPrice'] = data['avgRentPrice'].fillna((data['avgRentPrice'].mean()))
data['okumamis'] = data['okumamis'].fillna((data['okumamis'].mean()))
data['ilkokul'] = data['ilkokul'].fillna((data['ilkokul'].mean()))
data['ortaokul'] = data['ortaokul'].fillna((data['ortaokul'].mean()))
data['lise'] = data['lise'].fillna((data['lise'].mean()))
data['univ'] = data['univ'].fillna((data['univ'].mean()))
data['evli'] = data['evli'].fillna((data['evli'].mean()))
data['dul'] = data['dul'].fillna((data['dul'].mean()))
data['bosanmis'] = data['bosanmis'].fillna((data['bosanmis'].mean()))
data['bekar'] = data['bekar'].fillna((data['bekar'].mean()))
data['three'] = data['three'].fillna((data['three'].mean()))

city_mapping = {'İstanbul':'Marmara','Kocaeli':'Marmara', 'Manisa':'Ege','Izmir':'Ege','Aydin':'Ege',
                'Denizli':'Ege','Antalya':'Akdeniz','Adana':'Akdeniz','Afyon':'Ege','Bursa':'Marmara','Mersin':'Akdeniz','Isparta':'Akdeniz',
                'Kayseri':'Icand','Eskisehir':'Icand', 'Kütahya':'Ege',
                'Konya':'Icand','Ankara':'Icand','Samsun':'Karadeniz','Gaziantep':'Doguand'}


data['il'] = data['il'].map(city_mapping)
data=pd.get_dummies (data, columns= ['Gender'], drop_first=True)
data=pd.get_dummies (data, columns= ['il'], drop_first=True)

data=data.drop('ilce', axis=1)
data=data.drop('gsm',axis=1)

#data=data+1
#for i in data.columns:
#    data[i],_=stats.boxcox(data[i])


y=data['three']
X=data.drop(['three'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_test=X_test.as_matrix()
X_train=X_train.as_matrix()
y_train=y_train.as_matrix()
y_test=y_test.as_matrix()


array=cosine_similarity(X_train,X_test)

y_pred=[]

i=0

toplam=0

for i in range(3402):
     df=pd.DataFrame(data=array[:,2])
     ix=df.loc[df[0]==df.max().values[0]].index
     for k in ix:
         toplam=toplam+y_train[k]
     y_pred.append(toplam/len(ix))         
    #y_pred.append(y_train[i]*ix)



mse(y_test, y_pred)
r2(y_test,y_pred)