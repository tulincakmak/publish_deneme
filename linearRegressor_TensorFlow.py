# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import convertObject as cn
import sys

dataset = pd.read_csv('Churn_Modelling.csv')


#label = sys.argv[1]
#label=cn.convertObject(dataset, label)


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import pandas as pd
# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": X_train}),
    y=pd.Series(y_train),
    )


feature_columns = [tf.feature_column.numeric_column(X,shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=X)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

estimator.fit(input_fn=train_input_fn)
estimator.evaluate(input_fn=eval_input_fn)
estimator.predict(X=X)

tf.metrics.accuracy(labels=tf.argmax(y, 0), predictions=tf.argmax(y_test,0))
tf.metrics.precision(labels=y_test, predictions=X_test)
tf.metrics.confusion_matrix()
tf.metrics.mean_squared_error(labels=y_test, predictions=X_test)