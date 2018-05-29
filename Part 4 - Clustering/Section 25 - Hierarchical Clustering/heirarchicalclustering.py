
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using dendrogram for finding optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendogram')
plt.xlabel('customers')
plt/ylabel('euclidean distances')
plt.show()

#fit hc to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualzing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s =100, c='red', label = 'careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s =100, c='blue', label = 'standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s =100, c='green', label = 'target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s =100, c='cyan', label = 'carefree')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s =100, c='magenta', label = 'sensible')
plt.title('clusters of customers')
plt.xlabel('income')
plt.ylabel('spending score')
plt.legend()
plt.show()