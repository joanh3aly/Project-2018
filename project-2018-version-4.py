'''
Experiment with cross-tabulation, to count the amount of varieties in each cluster. Reveals accuracy of clustering and scaling.
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/joanhealy1/documents/exercise-5-iris-data/data/iris.data.csv')
#print(df.head())
print(df)

sl = df.iloc[:,0]
pl = df.iloc[:,2]
df3 = pd.DataFrame({'sl': sl, 'pl': pl})

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=3)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)
# Fit the pipeline to samples
pipeline.fit(df3)

# Calculate the cluster labels: labels
labels = pipeline.predict(df3)


#Testing to make count how many of each iris variety there are in the dataset
iris_class = df[df['variety'] == 'Iris-virginica'].shape[0]
df[df.variety == 'Iris-versicolor'].shape[0]
print("iris_class ", iris_class) # 50-Iris-virginica, 50-Iris-setosa, 50-Iris-versicolor

# Create a DataFrame with labels and species as columns: df
df4 = pd.DataFrame({'labels':labels,'variety':df['variety']})

# Create crosstab: ct
ct = pd.crosstab(df4['labels'],df4['variety'])

# Display ct
print(ct)
