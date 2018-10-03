# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:43:33 2017

@author: kerimtumkaya
"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def convertObject(data, label):
    objectTypes = list(data.select_dtypes(include=['object']).columns)
    
    for i in range(0, len(objectTypes)):
        if objectTypes[i] != label:
            lb = LabelBinarizer()
            lb.fit(data[objectTypes[i]])
            classes = lb.classes_
            transformed = lb.transform(data[objectTypes[i]])
            df = pd.DataFrame(data=transformed, columns=classes)
            frame = [df, data]
            data = pd.concat(frame, axis=1)
        
        else:
            le = LabelEncoder()
            le.fit(data[objectTypes[i]])
            transformed = lb.transform(data[objectTypes[i]])
            data[label] = transformed
    return data