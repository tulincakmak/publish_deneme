# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:42:05 2018

@author: tulincakmak
"""

import pandas as pd

data=pd.read_excel('vasdata.xlsx')

data.head()

#Create a new function:
def num_missing(x):
  return sum(x.isnull())

print ("Missing values per column:")
print (data.apply(num_missing, axis=0)) 

