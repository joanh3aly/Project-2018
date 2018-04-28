'''
Joan Healy - Project 2018 - Part 1 investigation
30/3/18

'''
# Import Python libraries 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import CSV data using Pandas library
df = pd.read_csv('/Users/joanhealy1/documents/exercise-5-iris-data/data/iris.data.csv')
# Print the first few rows of the dataset to examine
print(df.head())

# Find mean, standard deviation, minimum and maximum values of the different columns (representing length and width of sepals and petals)
sepal_length = df.sepal_length
print(np.mean(sepal_length))
print(max(sepal_length))
print(min(sepal_length))
print(np.std(sepal_length))

sepal_width = df.sepal_width
print(np.mean(sepal_width))
print(max(sepal_width))
print(min(sepal_width))
print(np.std(sepal_length))

petal_length = df.petal_length
print(np.mean(petal_length))
print(max(petal_length))
print(min(petal_length))
print(np.std(petal_length))

petal_width = df.petal_width
print(np.mean(petal_width))
print(max(petal_width))
print(min(petal_width))
print(np.std(petal_width))

# Plot on graph
plt.figure()
# Use scatterplot to compare sepals and petals
plt.plot(sepal_length, sepal_width )
plt.scatter(sepal_length, sepal_width )

# Visualise the different lengths and widths of sepals/petals on a bar chart
x = np.arange(4)
stds = [np.std(sepal_length), np.std(sepal_width), np.std(petal_length), np.std(petal_width)]
plt.bar(x, stds)
plt.xticks(rotation='vertical')
plt.show()





