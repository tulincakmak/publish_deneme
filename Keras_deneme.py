# -*- coding: utf-8 -*-
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

LABEL = "Exited"
dataset.drop("Surname", axis=1, inplace=True)
dataset.drop("RowNumber", axis=1, inplace=True)
dataset.drop("CustomerId", axis=1, inplace=True)
dataset.drop("Geography", axis=1, inplace=True)
dataset.drop("Gender", axis=1, inplace=True)

def split_data(data, rate, label):
    data = data.dropna()

    train_data, test_data = train_test_split(data, test_size=rate)

    train_label = train_data[label]
    train_data = train_data.drop(label, 1)

    test_label = test_data[label]
    test_data = test_data.drop(label, 1)
    return train_data, train_label, test_data, test_label

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

x_train, y_train, x_test, y_test = split_data(dataset, 0.20, LABEL)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


input_dim=len(x_train[0, :])
# Part 2 - Now let's make the ANN!



# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print(y_pred)

classifier.save('my_model.h5')  #BU KISIMDA MODELİ KAYDEDİYOR DOSYA UZANTISI H5 OLMALI BUNUN İÇİNDE ANACONDADANIN ENVİRONMENT ında DA YÜKLÜ OLMALI!!!
from keras.models import load_model
model=load_model('my_model.h5')
t = model.predict(x_test, batch_size=10, verbose=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(len(y_pred.round()))
print(cm)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier1=Sequential()
    classifier1.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=8))
    classifier1.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier1.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier1
classifier1=KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies=cross_val_score(estimator=classifier1, X=x_train, y=y_train, cv=10)
mean=accuracies.mean()
variance=accuracies.std()

ev=classifier.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], ev[1]*100))

print('accuracies = %s '% accuracies)
print('mean = %s '% mean)
print('variance = %s '% variance)
print('ev = %s '% ev)


#FİNDİNG BEST PARAMETERS FOR MODEL!!!!!

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier(optimizer):
#    classifier=Sequential()
#    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=8))
#    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
#    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
#    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#    return classifier
#classifier=KerasClassifier(build_fn=build_classifier)
#parameters={'batch_size': [25,32],
#            'epochs': [100, 500],
#            'optimizer': ['adam', 'rmsprop']
#        }
#grid_search=GridSearchCV(estimator=classifier,
#                         param_grid=parameters,
#                         scoring='accuracy',
#                         cv=10)
#grid_search=grid_search.fit(x_train, y_train)
#best_parameters=grid_search.best_params_
#best_accuracy=grid_search.best_score_


print(x_train[ : , 0])

