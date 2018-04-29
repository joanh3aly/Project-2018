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
# Find mean, standard deviation, minimum and maximum values of the different columns (representing length and width of sepals and petals)
sepal_length = df.sepal_length
print('sepal length mean {}, \n sepal length max {},\n sepal length min {},\n sepal length standard deviation {}'.format( np.mean(sepal_length), max(sepal_length), min(sepal_length), np.std(sepal_length) ))

sepal_width = df.sepal_width
print('sepal width mean {},\n sepal width max {},\n sepal width min {},\n sepal width standard deviation {}'.format( np.mean(sepal_width), max(sepal_width), min(sepal_width), np.std(sepal_width) ))

petal_length = df.petal_length
print('petal length mean {},\n petal length max {},\n petal length min {},\n petal length standard deviation {}'.format( np.mean(petal_length), max(petal_length), min(petal_length), np.std(petal_length) ))

petal_width = df.petal_width
print('petal width mean {},\n petal width max {},\n petal width min {},\n petal width standard deviation {}'.format( np.mean(petal_width), max(petal_width), min(petal_width), np.std(petal_width) ))

# Plot on graph
plt.figure()

# .plot creates a line graph (not great for visualising this type of data) 
#plt.plot(sepal_length, sepal_width )

# Scatterplot to compare sepals and petals (uncomment to run)
#plt.scatter(sepal_length, sepal_width )

# Visualise the different lengths and widths of sepals/petals on a bar chart
x = np.arange(4)
stds = [np.std(sepal_length), np.std(sepal_width), np.std(petal_length), np.std(petal_width)]
plt.bar(x, stds)
plt.xticks(rotation='vertical')

# Show plot
plt.show()





