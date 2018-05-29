#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:38:09 2018

@author: v0y000e
"""

#data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:,3].values

#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])
 
#encodingcategorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
x[:,0] = le_x.fit_transform(x[:,0])
ohe = OneHotEncoder(categorical_features =[0])
x = ohe.fit_transform(x).toarray()
le_y = LabelEncoder()
y = le_y.fit_transform(y)

#split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
