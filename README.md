# Project-2018
by Joan Healy

## About This Project

This is a project for my Python programming module for the GMIT HDip in the science of data analytics. It investigates and attempts to classify the 3 varieties of Iris flower using machine learning algorithms. I used several Python libraries and machine learning tutorials to achieve this, such as the Datacamp Unsupervised learning tutorial, which are outlined below. The aim of this project was to learn some basic unsupervised learning techniques.

## Summary of Fisher's Iris Data Set 

Fisher's Iris flower data set is a data set comprised of the length and width of the sepals and petals of 50 samples each of Iris setosa, Iris verginica and Iris versicolor, measured in centimeters. 

In 1936 Ronald Fisher, a British biologist and statistician used the data set to create a linear discriminant model to distinguish the Iris species from each other. His research is recorded in his paper "The use of multiple measurements in taxonomic problems".

## Initial Investigations – _project-2018-part-1.py_

The pandas library was used to import the dataset into my Python script. 
Numpy was used to calculate the mean, standard deviation, minimum and maximum values of the different columns (representing length and width of sepals and petals)
Mathplotlib was used to create scatterplots and bar charts comparing the different features: compare sepals and petals.

_Sample of the data using .head()_  

| index        | sepal_length           | sepal_width  | petal_length        | petal_width           | variety  |
| :-------------: |:-------------:| :-----:| :-------------: |:-------------:| :-----:|
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
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/standard-deviations-barchart.png "Barchart")


## KMeans clustering and visualisation of centroids – _project-2018-part-2.py_
The KMeans algorithm from Sklearn was used to model the dataset after the petal length and sepal length columns were put into a new dataframe. Labels for each were found by using .fit and .predict on the model.
From this visualisation, we can see 3 clusters of data points.

_Scatterplot of clustering centroids found after modelling the petal length and sepal length using KMeans_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/centroid-clusters.png "Centroids")

## Finding the best number of clusters by visualising the inertia – _project-2018-part-3.py_
Inertia is a measure of the sum of the squares within each cluster. The lower the inertia of each cluster, the more accurate each cluster will be. The .inertia_ method was used on a range of clustering values (6 and 20 in this case), and then visualised as a line graph. This information can be used to determine how many clusters to use in the model. In the case of the Iris data set, it is 3, as can be seen from the graph below.

_Line graph of inertia vs 6 clusters_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/inertia.png "Inertia")

_Line graph of inertia vs 20 clusters_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/inertia-20.png "Inertia 20")

## Tranformation of data features using Scaler & Normalizer to improve accuracy, and use of a crosstabulation table to test accuracy – _project-2018-part-4.py_
The data in each feature was transformed using the scaler and normalizer functions from sklearn.preprocessing. Scaler standardises the variance of the data in each feature, Normalizer rescales each row in the dataset so that the norm equals 1. 
A cross-tabulation table is created for each model using Normalizer, Scaler and no data transformations. Each model's accuracy can then be compared.


_3 clusters - no transformation of the features_
| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| ------------- |:-------------:| -----:| ------------- |
|0  | 0          | 4        | 37         | 
|1  | 50         | 1         | 0         | 
|2  | 0          | 45        | 13         | 


_3 clusters with Scaler_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| ------------- |:-------------:| -----:| ------------- |
|0  | 50          | 4        | 0         | 
|1  | 0         | 9         | 34         | 
|2  | 0          | 37        | 16         | 

_3 clusters with Normalizer_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| ------------- |:-------------:| -----:| ------------- |
|0  | 0          | 9       | 49        | 
|1  | 50         | 0         | 0         | 
|2  | 0          | 41        | 1         | 

_3 clusters with scaler and normalizer_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| ------------- |-------------:| -----:| ------------- |
|0  | 0          | 17       | 6        | 
|1  | 50         | 7         | 0         | 
|2  | 0          | 26        | 44         | 


## Summary of investigations and further research 


## References:
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://matplotlib.org/gallery/specialty_plots/system_monitor.html#sphx-glr-gallery-specialty-plots-system-monitor-py  
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html


