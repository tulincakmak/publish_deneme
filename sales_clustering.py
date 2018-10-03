

from sklearn.cluster import KMeans
from sklearn.cluster import k_means
import numpy as np
import pandas as pd
from sklearn import metrics


data = pd.ExcelFile("C:\\Users\\AhmetTezcanTekin\\Desktop\\cluster.xlsx")


mapping={'Low':1, 'Below Average':2, 'Average':3, 'Above Average':4, 'High':5}
data['booking_value_index']=data['booking_value_index'].map(mapping)

most=data['hotel_types'].value_counts()
data['hotel_types']=data['hotel_types'].fillna(most.index[0])

mapping2={'Summer': 0, 'City': 1}
data['hotel_types']=data['hotel_types'].map(mapping)

data['hotel_types'].unique()



X = np.array(df3)


kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.labels_



kmeans.predict(X)

kmeans.cluster_centers_

labels=kmeans.labels_



print(labels)

print(X)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

