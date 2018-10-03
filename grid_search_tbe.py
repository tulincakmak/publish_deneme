# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:39:29 2018

@author: tulincakmak
"""

import pandas as pd
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
import numpy as np


data5=pd.read_excel('CaglaExcel.xlsx')

#data5=pd.read_csv('CaglaExcel3.csv',engine="python",sep=None)

data5['Bölge'] = data5['Bölge'].str.strip()
data5['booking_value_index'] = data5['booking_value_index'].str.strip()

data5.groupby('Bölge')['Bölge'].count()
data5['Bölge']=data5['Bölge'].fillna('Marmara')

data5["Bölge"]=data5["Bölge"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))


mapping2={'Marmara': 'Marmara', 'Ege': 'Ege', 'Akdeniz':'Akdeniz', 'Karadeniz':'Others', 'İçAnadolu':'Others', 'DoğuAnadolu': 'Others', 'GüneydoğuAnadolu': 'Others'}
data5['Bölge']=data5['Bölge'].astype(str).map(mapping2)

data5=pd.get_dummies (data5, columns= ['Bölge'], drop_first=False)

data5["booking_value_index"]=data5["booking_value_index"].astype(str).apply(lambda x: bytes(x, "utf-8").decode("unicode_escape").replace("\t", "").replace("\n", "").replace("\r\n",""))

mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data5['booking_value_index']=data5['booking_value_index'].map(mapping)

data5=pd.get_dummies (data5, columns= ['booking_value_index'], drop_first=False) 

data5=data5.drop('trivagoId', axis=1)

data5['rating'] = data5['rating'].fillna(data5['rating'].mean)

to_be_replaced = ['beat2','beat3','beat4','beat5','beat6','beat7','beat8','meet2','meet3','meet4','meet5','meet6','meet7','meet8','lose2','lose3',
                 'lose4','lose5','lose6','lose7','lose8','outbid_ratio2','outbid_ratio3','outbid_ratio4','outbid_ratio5','outbid_ratio6',
                 'outbid_ratio7','outbid_ratio8',
                 'top_pos_share2','top_pos_share3','top_pos_share4','top_pos_share5','top_pos_share6','top_pos_share7','top_pos_share8']
                
data5[to_be_replaced] = data5[to_be_replaced].replace(10.0, 1.0)
data5[to_be_replaced] = data5[to_be_replaced].replace(10000.0, 1.0)


def missing(x):
    return sum(x.isnull())

print (data5.apply(missing, axis=0))



#correlation matrix
def correlation_matrix(X):
   from matplotlib import pyplot as plt
   from matplotlib import cm as cm

   fig = plt.figure(figsize=(16, 12))
   ax1 = fig.add_subplot(111)
   cmap = cm.get_cmap('jet', 30)
   cax = ax1.imshow(data5.corr(),  cmap=cmap)
   ax1.grid(True)
   plt.title('OtelZ Feature Correlation')
   labels=data5.columns
   ax1.set_xticks(range(len(labels)))
   ax1.set_yticks(range(len(labels)))
   ax1.set_xticklabels(labels,fontsize=8, rotation=45)
   ax1.set_yticklabels(labels,fontsize=8)
   # Add colorbar, make sure to specify tick locations to match desired ticklabels
   fig.colorbar(cax, ticks=[-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
   plt.gcf().subplots_adjust(bottom=0.15)
   plt.show()
   fig.savefig('to2.png')
   plt.close(fig)
  

corr_mat = correlation_matrix(data5)
from scipy import stats
from scipy.stats import gaussian_kde


def calc_MI(x, y, bins):
  c_xy = np.histogram2d(x, y, bins)[0]
  mi = mutual_info_score(None, None, contingency=c_xy)
  return mi

for i in data5.columns:
    calc_MI(data5['bid'],data5[i],bins=100)

for k in data5.columns:
    print( k, stats.spearmanr(data5[k],data5['clicks']))


xgb=XGBRegressor(learning_rate=0.1, max_depth =8, n_estimators=200)


from sklearn.model_selection import train_test_split
X=data5.drop('clicks', axis=1)
y=data5['clicks']



#def hotel_based_split(grouped, test_rate, seed):
#    """ grouped is a groupby dataframe object """
#    ts = grouped.apply(lambda x : x.sample(frac=test_rate, random_state=seed))
#    ts.index = ts.index.levels[1]
#    tr = grouped.drop(ts.index)
#    return tr, ts
#
#data2 = data.groupby('trivagoId')
### 1 örnek
#train_data, test_data = hotel_based_split(data2.copy(), 0.25, 333)
#train_data, val_data = hotel_based_split(train_data, 0.1875, 333)



n_estimators = [50, 100, 150, 200]
max_depth = [2, 3, 4, 6, 7, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)





best={}
def grid (model, train, valid, seed):
        grid_search = GridSearchCV(xgb, param_grid, scoring="r2", n_jobs=-1,  verbose=1)
        grid_result = grid_search.fit(train, valid)
        best['best_params '+str(seed)]=grid_result.best_params_
        best['best_score '+str(seed)]=grid_result.best_score_

    
best


bootstrapped_data = {}
seed=1
for seed in range(10):
     X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25, random_state=seed, shuffle=True)
     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed, shuffle=True) 
     bootstrapped_data['train_data'+str(seed)] = X_train
     bootstrapped_data['val_data'+str(seed)] = X_val
     bootstrapped_data['test_data'+str(seed)] = X_test   
     grid(xgb, X_val, y_val, seed)
    
    



