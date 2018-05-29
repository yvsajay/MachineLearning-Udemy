# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#remove dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ANN

#importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialize ann
classifier = Sequential()

#adding input layer and first hidden layer
classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))

#adding 2nd hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))

#adding output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer =  'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #adam is stochastic gradient descent
#loss if binary, then logarthmic loss function is binary_crossentropy
#loss if more than 2, then logarthmic loss function is categorical_crossentropy

#fitting ann to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100 )

#predicting and evaluating model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
