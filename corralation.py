# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:44:14 2018

@author: tulincakmak
"""


import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel('check_correlation2.xlsx')

corr = data.corr()
plt.figure(figsize=(14,8))
plt.title('Overall Correlation of CC', fontsize=18)
sns.heatmap(corr,annot=False,cmap='BrBG',linewidths=0.2,annot_kws={'size':20})
plt.show()

labels=data.columns.values

for feature in labels:
    corr, p_value = pearsonr(data[feature], data['avg_cpc'])
    print(feature,corr,p_value)
    
for feature in labels:
    corr, p_value = spearmanr(data[feature], data['avg_cpc'])
    print(feature,corr,p_value)
    
data=data.drop('net_total_cost', axis=1) 
data.info()

dropped=['booking_value_index', 'region', 'status']
data=data.drop(dropped, axis=1)
data.info()

# Correction Matrix Plot
import numpy
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = data.columns
#data = pandas.read_csv(url, names=names)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,54,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


from sklearn.metrics import r2_score as r2

r2_score=r2(data['onnisan_clicks'],data['Prediction_clicks'])
print(r2_score)


from sklearn.metrics import mean_squared_error as mse
mse=mse(data['onnisan_clicks'],data['Prediction_clicks'])
print(mse)