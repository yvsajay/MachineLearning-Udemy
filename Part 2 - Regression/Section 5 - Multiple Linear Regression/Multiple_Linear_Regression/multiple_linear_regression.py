# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('gmv _nps_age_seller.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

''''
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]
''''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = y_train.reshape(-1, 1)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

y_new_inverse = sc_y.inverse_transform(y_pred)

#building optimal model using backward elimination
import statsmodels.formula.api as sm 
X_train = np.append(arr = np.ones((224,1)).astype(int), values = X_train, axis = 1) 
X_opt = X_train[:, [0,2]]
rg_ols = sm.OLS(endog = y_train, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()