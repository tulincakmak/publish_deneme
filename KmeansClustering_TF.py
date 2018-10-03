# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np


CSV_COLUMNS =  ["RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts",
           "HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
#["RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age","Tenure","Balance",
               #"NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]

FEATURES = ["RowNumber","CustomerId","CreditScore","Age","Tenure","Balance","NumOfProducts",
           "HasCrCard","IsActiveMember", "EstimatedSalary"]
#["RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age","Tenure","Balance",
               #"NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
drop_cols=["RowNumber","Surname","Geography","Gender"]

LABEL="Exited"

df_train = pd.read_csv("Churn_Modelling.csv", names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
df_test = pd.read_csv("Churn_Modelling.csv", names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)


#Bu fonksiyonda test ve label olarak ayırıyor????
"""def get_input_fn(data_set, num_epochs=None, shuffle=False):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle,
      batch_size=100)
""" 
  
df_train = pd.read_csv('Churn_Modelling.csv',  names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
df_train.drop(drop_cols, axis = 1, inplace=True)

df_test = pd.read_csv('Churn_Modelling.csv',  names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
df_test.drop(drop_cols, axis = 1, inplace=True)
#print(df_train.head(5))

#Bir tane de evaluate için yazılacak!!  
def input_fn():
  return tf.constant(df_train.as_matrix(), tf.float32, df_train.shape), None


def input_fn_test():
  return tf.constant(df_test.as_matrix(), tf.float32, df_train.shape), None

tf.logging.set_verbosity(tf.logging.ERROR)

##Burdaki cluster dinamik alınacak. Labeldaki distinct eleman sayısına eşit olmalı??????
kmeans = tf.contrib.learn.KMeansClustering(
    num_clusters=2, relative_tolerance=0.0001)


kmeans.fit(x=None,y=None,input_fn=input_fn)

clusters = kmeans.clusters()

assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn_test))

#print(assignments)

rowNumber=0
for i in assignments:
    print(str(rowNumber)+': '+str(assignments[i]))
    rowNumber=rowNumber+1


kmeans.score(input_fn=input_fn)

kmeans.evaluate(input_fn=input_fn_test)


kmeans.SQUARED_EUCLIDEAN_DISTANCE(input_fn=input_fn_test)


def calculate(prediction, LABEL):
    arr = {"accuracy": tf.metrics.accuracy(labels=LABEL, predictions=prediction),
           "MSE": tf.metrics.root_mean_squared_error(labels=LABEL,predictions=prediction),
           "precision": tf.metrics.precision(labels=LABEL,predictions=prediction),
           "AUC": tf.metrics.auc(labels=LABEL,predictions=prediction),
           "True_Positive_Rates": tf.metrics.true_positives(labels=LABEL,predictions=prediction),
           "False_Negative_Rates": tf.metrics.false_negatives(labels=LABEL,predictions=prediction),
           "False_Positive_Rates": tf.metrics.false_positives(labels=LABEL,predictions=prediction),
           "True_Negative_Rates": tf.metrics.true_negatives_at_thresholds(labels=LABEL,predictions=prediction),
           "Confusion_Matrix": tf.confusion_matrix(labels=LABEL,predictions=prediction)
           }
    return arr
 

calculate(df_test[LABEL],assignments)


#Bu kısıma bakılacak!!!!!!
"""import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def ScatterPlot(X, Y, assignments=None, centers=None):
  if assignments is None:
    assignments = [0] * len(X)
  fig = plt.figure(figsize=(10,5))
  cmap = ListedColormap(['red', 'green'])
  plt.scatter(X, Y, c=assignments, cmap=cmap)
  if centers is not None:
    plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), 
                marker='+', s=100, cmap=cmap)  
  plt.xlabel('Exited')
  plt.ylabel('Age')
  plt.show()


ScatterPlot(df_train.Exited, df_train.Age, assignments, clusters)
"""
