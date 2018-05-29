# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)""" 

#fitting lr to dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


#fitting plr to dataset
from sklearn.preprocessing import PolynomialFeatures
plr=  PolynomialFeatures(degree = 4)
X_poly = plr.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_poly,y)


#visalizing lr results
plt.scatter(X,y, color='red')
plt.plot(X,lr.predict(X), color = 'blue')
plt.title('truth or bluff (lr)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualizing plr results
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X,lr2.predict(plr.fit_transform(X)), color = 'blue')
plt.title('truth or bluff (plr)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predict newresult with lr
lr.predict(6.5)

#predict new result with plr
lr2.predict(plr.fit_transform(6.5))