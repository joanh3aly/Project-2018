'''
Project 2018 - part 4
Joan Healy
22/4/18

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

# Create scaler
scaler = StandardScaler()
# Create a normalizer
normalizer = Normalizer()

# Create KMeans instance
kmeans = KMeans(n_clusters=3)

# Create pipeline with kmeans
pipeline = make_pipeline(kmeans)
# Create pipeline with scaler & kmeans (uncomment to run)
#pipeline = make_pipeline(scaler,kmeans)
# Create pipeline with normalizer & kmeans (uncomment to run)
#pipeline = make_pipeline(normalizer,kmeans)
# Create pipeline with both scaler + normalizer & kmeans (uncomment to run)
#pipeline = make_pipeline(normalizer, scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(df3)

# Calculate the cluster labels
labels = pipeline.predict(df3)

# Test to count how many of each iris variety there are in the dataset
iris_class = df[df['variety'] == 'Iris-virginica'].shape[0]
print("iris_class ", iris_class) 

# Create a DataFrame with labels and Iris variety as columns
df4 = pd.DataFrame({'labels':labels,'variety':df['variety']})

# Create crosstab
ct = pd.crosstab(df4['labels'],df4['variety'])

# Display ct
print(ct)
