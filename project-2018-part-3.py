'''
Joan Healy - Project 2018 - part 3
22/4/18

Objective: Find the best number of clusters for the model and graph output.
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('/Users/joanhealy1/documents/exercise-5-iris-data/data/iris.data.csv')
sl = df.iloc[:,0]
pl = df.iloc[:,2]
df3 = pd.DataFrame({'sl': sl, 'pl': pl})

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df3)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()





