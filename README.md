# Project-2018
by Joan Healy

## About This Project

This is a project for my Python programming module for the GMIT HDip in the science of data analytics. It investigates and attempts to classify the 3 varieties of Iris flower using unsupervised learning techniques. I used several Python libraries and machine learning tutorials to achieve this, which are outlined below. 

## Summary of Fisher's Iris Data Set 

Fisher's Iris flower data set is a data set comprised of the length and width of the sepals and petals of 50 samples each of Iris setosa, Iris verginica and Iris versicolor, measured in centimeters. 

In 1936 Ronald Fisher, a British biologist and statistician used the data set to create a linear discriminant model to distinguish the Iris species from each other. His research is recorded in his paper "The use of multiple measurements in taxonomic problems".

## Initial Investigations 

The pandas library was used to import the dataset into my Python script. 
Numpy was used to calculate the mean, standard deviation, minimum and maximum values of the different columns (representing length and width of sepals and petals)
Mathplotlib was used to create scatterplots and bar charts comparing the different features: compare sepals and petals.

_Sample of the data using .head()_  
|   | sepal_length   | sepal_width   | petal_length   | petal_width   | variety     |
|---|--------------|-------------|--------------|-------------|-------------|
|0  | 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa |
|1  | 4.9          | 3.0         | 1.4          | 0.2         | Iris-setosa |
|2  | 4.7          | 3.2         | 1.3          | 0.2         | Iris-setosa |
|3  | 4.6          | 3.1         | 1.5          | 0.2         | Iris-setosa |
|4  | 5.0          | 3.6         | 1.4          | 0.2         | Iris-setosa |

_Sepal Length_  
mean 5.843333333333335,   
max 7.9,   
min 4.3,  
standard deviation 0.8253012917851409  

_Sepal Width_  
mean 3.0540000000000007,   
max 4.4,   
min 2.0,   
standard deviation 0.4321465800705435  

_Petal Length_  
mean 3.7586666666666693,   
max 6.9,   
min 1.0,   
standard deviation 1.7585291834055201  

_Petal Width_   
mean 1.1986666666666672,   
max 2.5,   
min 0.1,   
standard deviation 0.760612618588172  

_Standard deviations of the 4 features in the data set:_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/standard-deviations-barchart.png "Logo Title Text 1")




## References:
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://matplotlib.org/gallery/specialty_plots/system_monitor.html#sphx-glr-gallery-specialty-plots-system-monitor-py  
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html


