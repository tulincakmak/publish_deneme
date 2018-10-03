

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics
from functools import reduce
import pyodbc

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
cursor = cnxn.cursor()


sql="exec dbo.get_data_for_clustering '2018-06-24'"
df = pd.read_sql(sql,cnxn)




mapping={'City Hotel':1, 'Summer Hotel':2}
df['hotel_types']=df['hotel_types'].map(mapping)

df= df.drop(['country','region'], axis=1)

df=df.fillna(0)


X = np.array(df)


kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.labels_

kmeans.predict(X)

kmeans.cluster_centers_

labels=kmeans.labels_

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))


labels=pd.DataFrame(data=labels)
labels2=labels.reset_index(drop=True)

trivagoId=pd.DataFrame(data=df['trivagoID'])
trivagoId2=trivagoId.reset_index(drop=True)

result=pd.concat([trivagoId2,labels2], axis=1)




#cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=Tempdata;UID=caglaS;PWD=c:agla12S')
#cursor = cnxn.cursor()
#
#
#sql = """CREATE TABLE cluster_2406 (
#          trivago_id INT,  
#          cluster INT )"""
#cursor.execute(sql)
#cnxn.commit()
#
#from sqlalchemy import create_engine
#engine = create_engine('mssql+pymssql://tulinC:tlnckmk@78.40.231.196/Tempdata', echo=False)
#result.to_sql(name='cluster_2406_2', con='mssql+pymssql://tulinC:tlnckmk@78.40.231.196/Tempdata', index=False)
#engine.execute("SELECT * FROM cluster_2406").fetchall()
#
#
#cnxn.close()


result.to_excel('cluster_2406.xlsx')