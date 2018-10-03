# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:20:24 2018

@author: tulincakmak
"""


import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy import stats
import pymrmr


data=pd.read_excel('everthing_avarage2_traindata.xlsx')

#test=pd.read_excel('17042018_otelz_test_data.xlsx')
data2=data.copy()

data['bölge'] = data['bölge'].str.strip()
data['booking_value_index'] = data['booking_value_index'].str.strip()

data.groupby('bölge')['bölge'].count()
data['bölge']=data['bölge'].fillna('Marmara')

data["bölge"]=data["bölge"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))


mapping2={'Marmara': 'Marmara', 'Ege': 'Ege', 'Akdeniz':'Akdeniz', 'Karadeniz':'Others', 'İçAnadolu':'Others', 'DoğuAnadolu': 'Others', 'GüneydoğuAnadolu': 'Others'}
data['bölge']=data['bölge'].astype(str).map(mapping2)

data=pd.get_dummies (data, columns= ['bölge'], drop_first=False)

data["booking_value_index"]=data["booking_value_index"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))

mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

data=pd.get_dummies (data, columns= ['booking_value_index'], drop_first=False) 

data=pd.get_dummies (data, columns= ['date_name'], drop_first=False) 

data=data.drop('trivagoId', axis=1)

data['rating'] = data['rating'].fillna(data['rating'].mean)

to_be_replaced = ['beat2','beat3','beat4','beat5','beat6','beat7','beat8','meet2','meet3','meet4','meet5','meet6','meet7','meet8','lose2','lose3',
                 'lose4','lose5','lose6','lose7','lose8','outbid_ratio2','outbid_ratio3','outbid_ratio4','outbid_ratio5','outbid_ratio6',
                 'outbid_ratio7','outbid_ratio8',
                 'top_pos_share2','top_pos_share3','top_pos_share4','top_pos_share5','top_pos_share6','top_pos_share7','top_pos_share8']
                
data[to_be_replaced] = data[to_be_replaced].replace(10.0, 1.0)
data[to_be_replaced] = data[to_be_replaced].replace(10000.0, 1.0)


def calc_MI(x, y, bins):
  c_xy = np.histogram2d(x, y, bins)[0]
  mi = mutual_info_score(None, None, contingency=c_xy)
  return mi

for i in data.columns:
    print(calc_MI(data['bid'],data[i],bins=100))

from scipy.stats import chi2_contingency

def calc_MI(x, y, bins):
   c_xy = np.histogram2d(x, y, bins)[0]
   g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
   mi = 0.5 * g / c_xy.sum()
   return mi

for i in X2_train.columns:
    calc_MI(X2_train[i], y2_train, 3) 
   
def missing(x):
    return sum(x.isnull())

print (data.apply(missing, axis=0))


for k in data.columns:
    print( k, stats.spearmanr(data[k],data['bid']))
    

from sklearn.model_selection import train_test_split
X2=data.drop('clicks', axis=1)
y2=data['clicks']

X2_train, X2_test, y2_train, y2_test  = train_test_split(X2, y2, test_size=0.25, random_state=4, shuffle=True)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.25, random_state=4, shuffle=True) 


#result=pd.concat([y2_train,X2_train], axis=1)
#
#columns=pymrmr.mRMR(data,'MIQ',25)
#
#print(columns)

#data3=data[columns]


X2_train=X2_train+1
for i in X2_train.columns:
    X2_train[i],_=stats.boxcox(X2_train[i])
    
    
    
X2_train=np.log(X2_train+1) 
y2_train=np.log(y2_train+1)

X2_val=np.log(X2_val+1) 
y2_val=np.log(y2_val+1)

X2_test=np.log(X2_test+1) 
y2_test=np.log(y2_test+1)     




from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)
xgb.fit(X=X2_train, y=y2_train)
y_pred=xgb.predict(X2_val)
mse_score=mse(y2_val,y_pred)
print(mse_score)
r2_score=r2(y2_val,y_pred)
print(r2_score)

cr=cross_val_score(xgb, X2_train, y2_train, cv=3)
print(cr.mean())

sfm=SelectFromModel(xgb)
sfm.fit(X2_train, y2_train)
x_train_important=sfm.transform(X2_train)
outcome=sfm.get_support()
for i in range (0,len(X2_train.columns)):
    if outcome[i]:
        print(X2_train.columns[i])

print(x_train_important)


 mse_score=mse(y2_test,y_pred)
 print(mse_score)
 r2_score=r2(y2_test,y_pred)
 print(r2_score)
