# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values



#encodingcategorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X[:,3] = le_x.fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features =[3])
X = ohe.fit_transform(X).toarray()

#avoiding dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fit mlr to training set
from sklearn.linear_model import LinearRegression
rg= LinearRegression()
rg.fit(X_train, y_train) 

#fit mlr to test set
y_pred = rg.predict(X_test)
 
#building optimal model using backward elimination
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 
X_opt = X[:, [0,1,2,3,4,5]]
rg_ols = sm.OLS(endog = y, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()
X_opt = X[:, [0,1,3,4,5]]
rg_ols = sm.OLS(endog = y, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()
X_opt = X[:, [0,3,4,5]]
rg_ols = sm.OLS(endog = y, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()
X_opt = X[:, [0,3,5]]
rg_ols = sm.OLS(endog = y, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()
X_opt = X[:, [0,3]]
rg_ols = sm.OLS(endog = y, exog = X_opt).fit() #ols = ordinary least square
rg_ols.summary()