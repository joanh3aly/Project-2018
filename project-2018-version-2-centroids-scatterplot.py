'''
Joan Healy - Project 2018
30/3/18

Objective: Use KMeans to cluster the data points of petal and sepal length, then scatterplot to show centroids (centers of each cluster).
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('/Users/joanhealy1/documents/exercise-5-iris-data/data/iris.data.csv')
#print(df.head())

# Experimenting with various ways to manipulate the dataframe using .iloc and .drop methods.
#points = df.drop(['class',], axis=1)
sl = df.iloc[:,0]
pl = df.iloc[:,2]
#df2.loc[:,'a':'b'] = p.Series(np.random.randn(sLength), index=df1.index)
df3 = pd.DataFrame({'sl': sl, 'pl': pl})

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(df3)

# Determine the cluster labels of new_points: labels
labels = model.predict(df3)

# Print cluster labels of new_points
print(labels)

# Assign the columns of new_points: xs and ys
print(df3.iloc[:,0])
xs = df3.iloc[:,0]
ys = df3.iloc[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()





