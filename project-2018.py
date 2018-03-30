'''
Joan Healy - Project 2018
30/3/18

1. Research background information about the data set and write a summary about it.
2. Keep a list of references you used in completing the project.
3. Download the data set and write some Python code to investigate it.
4. Summarise the data set by, for example, calculating the maximum, minimum and mean of each column of the data set. A Python script will quickly do this for you.
5. Write a summary of your investigations.
6. Include supporting tables and graphics as you deem necessary.

TODO:
? Figure out which species is which by the data alone - supervised learning ?
Build into functions/classes as appropriate.
https://matplotlib.org/gallery/specialty_plots/system_monitor.html#sphx-glr-gallery-specialty-plots-system-monitor-py
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/joanhealy1/documents/exercise-5-iris-data/data/iris.data.csv')
print(df.head())

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

petal_length = df.sepal_length
petal_width = df.petal_width

# Plot on graph
plt.figure()
#plt.plot(sepal_length, sepal_width )
#plt.scatter(sepal_length, sepal_width )
x = np.arange(4)
stds = [np.std(sepal_length), np.std(sepal_width), np.std(petal_length), np.std(petal_width)]
plt.bar(x, stds)
plt.xticks(rotation='vertical')

plt.show()





