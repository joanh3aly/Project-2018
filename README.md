# Project-2018
by Joan Healy

## About This Project

This is a project for my Python programming module for the GMIT HDip in the science of data analytics. It investigates and attempts to classify the 3 varieties of Iris flower in Fisher's Iris Dataset based upon clustering petal length and width using machine learning algorithms. I used several Python libraries and machine learning tutorials to achieve this, such as the [Datacamp Unsupervised learning tutorial](https://campus.datacamp.com/courses/unsupervised-learning-in-python/clustering-for-dataset-exploration?ex=1), which are outlined below. The aim of this project was to learn some basic unsupervised learning techniques.

## Summary of Fisher's Iris Data Set 

Fisher's Iris flower data set is a data set comprised of the length and width of the sepals and petals of 50 samples each of Iris setosa, Iris verginica and Iris versicolor, measured in centimeters. 

In 1936 Ronald Fisher, a British biologist and statistician used the data set to create a linear discriminant model to distinguish the Iris species from each other. His research is recorded in his paper "The use of multiple measurements in taxonomic problems".

## Initial Investigations – _project-2018-part-1.py_

The pandas library was used to import the dataset into my Python script. 
Numpy was used to calculate the mean, standard deviation, minimum and maximum values of the different columns (representing length and width of sepals and petals).  

Mathplotlib was used to create scatterplots and bar charts comparing the different features: width and length of sepals and petals. Petal length vs petal width shows two clear clusters of data points with a small cluster on the bottom left and a larger cluster stretching from the middle of the graph to the top right corner. We can find less defined clusters for sepal width vs sepal length. Sepal length vs petal width and sepal width vs petal length have defined clusters but are possibly not as effective as to classify as these are different parts of the flower.

The bar chart of the standard deviation of the different features shows a great difference in the variance of the data, with petal length being of high variance and sepal width being the lowest. 

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

_Petal length vs petal width_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/petal_length_vs_petal_width.png "petal_length_vs_petal_width")

_Sepal length vs petal length_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/sepal_length-vs-petal_length.png "sepal_length-vs-petal_length")

_Sepal width vs petal width_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/sepal_width_vs_petal_width.png "sepal_width_vs_petal_width")

_Sepal width vs sepal length_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/sepal_width_vs_sepal_length.png "sepal_width_vs_sepal_length")


## KMeans clustering and visualisation of centroids – _project-2018-part-2.py_
The KMeans algorithm from Sklearn was used to model the dataset after the petal length and sepal length columns were put into a new dataframe. The Kmeans algorithm was chosen as it is a classic model for pattern recognition. Unsupervised learning algorithms learn by themselves, we don't need to train them (Yang, 2013, p12). Labels for each were found by using .fit and .predict on the model.   

From this visualisation, we can see 3 clusters of data points. Based upon our initial investigations, we can see the smaller leaved Iris variety cluster in the lower left corner, and the 2 larger varieties merging between 3 and 7mm on the x-axis. This visualisation is similar to the petal length vs petal width scatterplot above. [TODO] Possible bug, in that I'm not sure why the range of values on the y-axis is different to the scatterplot (goes to 8mm) when the max petal width is 2.5mm. 

_Scatterplot of clustering centroids found after modelling the petal length and sepal length using KMeans_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/centroid-clusters.png "Centroids")

## Finding the best number of clusters by visualising the inertia – _project-2018-part-3.py_
Inertia is a measure of the sum of the squares within each cluster [Sklearn documentation](http://scikit-learn.org/stable/modules/clustering.html). The lower the inertia of each cluster, the more accurate each cluster will be. The .inertia_ method was used on a range of clustering values (6 and 20 in this case), and then visualised as a line graph. This information can be used to determine how many clusters to use in the model. In the case of the Iris data set, it is 3, as can be seen from the graph below.

_Line graph of inertia vs 6 clusters_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/inertia.png "Inertia")

_Line graph of inertia vs 20 clusters_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/inertia-20.png "Inertia 20")

## Tranformation of data features using Scaler & Normalizer to improve accuracy, and use of a crosstabulation table to test accuracy – _project-2018-part-4.py_
The data in each feature was transformed using the scaler and normalizer functions from sklearn.preprocessing. Scaler standardises the variance of the data in each feature, Normalizer rescales each row in the dataset so that the norm equals 1. 
A cross-tabulation table is created for each model using Normalizer, Scaler and no data transformations. Each model's accuracy can then be compared.


_3 clusters - no transformation of the features_  

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| :-------------: |:-------------:| :-----:| :-------------: |
|0  | 0          | 4        | 37         | 
|1  | 50         | 1         | 0         | 
|2  | 0          | 45        | 13         | 


_3 clusters with Scaler_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| :------------- |:-------------:| :-----:| :-------------: |
|0  | 50          | 4        | 0         | 
|1  | 0         | 9         | 34         | 
|2  | 0          | 37        | 16         | 

_3 clusters with Normalizer_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| :------------- |:-------------:| :-----:| :-------------: |
|0  | 0          | 9       | 49        | 
|1  | 50         | 0         | 0         | 
|2  | 0          | 41        | 1         | 

_3 clusters with scaler and normalizer_

| labels        | Iris-setosa           | Iris-versicolor  | Iris-virginica    | 
| :-------------: |:-------------:| :-----:| :------------- |
|0  | 0          | 17       | 6        | 
|1  | 50         | 7         | 0         | 
|2  | 0          | 26        | 44         | 


## Summary of investigations and further research 
As we can see from the 4 cross-tabulation tables above, the best model for the data used 3 clusters and the Normalizer preprocessor of the feature data.   

* The number of clusters corresponds to what we know about the dataset, in that there are 3 varieties of Iris present. Using the inertia method to test the number of clusters further establishes this.

* The Normalizer preprocessor cross-tabulation table shows us that it was the most accurate preprocessor, over Scaler and no preprocessing, as we can see that Iris setosa and Iris Virginica are at 50 samples, or almost 50 (49). Only Iris versicolor is a little less accurate, yet still more accurate compared to the other tables. This is perhaps because there is quite a difference between the standard deviation of the petal length feature and the sepal length feature at 1.75 and 0.4 respectively (see bar chart below). Therefore normalizing them makes them easier for Kmeans to compare.

_Standard deviations of the 4 features in the data set:_
![alt text](https://github.com/joanh3aly/Project-2018/blob/master/figures/standard-deviations-barchart.png "Barchart")

_Further Research_

* Even though the clustering algorithm correctly identified (almost) 50 of each species, how do I know that the algorithm correctly identified each species at the granular level? For example, how do we know if there were false positives or false negatives?
* Why did scaler and normalizer working together not make the algorithm more accurate is spotting patterns?
* Would the model be more/less accurate if different features were used - ie: sepal width vs petal width, or sepal length vs sepal width?


## References:
[UC Irvine Machine Learning Repository. Iris data set.](http://archive.ics.uci.edu/ml/datasets/Iris)
[Datacamp Unsupervised learning tutorial](https://campus.datacamp.com/courses/unsupervised-learning-in-python/clustering-for-dataset-exploration?ex=1)  
[Iris Flower Dataset, Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)  
[Udacity Machine Learning](https://eu.udacity.com/course/intro-to-machine-learning--ud120)  
[Sklearn](http://scikit-learn.org/)  
[Pandas dataframe documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)  
[Numpy statistics functions documentation](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.statistics.html)  
[Mathplotlib bar chart](https://matplotlib.org/gallery/specialty_plots/system_monitor.html#sphx-glr-gallery-specialty-plots-system-monitor-py)  
[Mathplotlib scatterplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)  
[Yang, Yu, 2013, "A study of pattern recognition of Iris
flower based on Machine Learning "](https://www.theseus.fi/bitstream/handle/10024/64785/yang_yu.pdf?sequence=1&isAllowed=y)
[Minitab - Interpret all statistics and graphs for Cluster K-Means](https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/multivariate/how-to/cluster-k-means/interpret-the-results/all-statistics-and-graphs/)  
[Sklearn documentation](http://scikit-learn.org/stable/modules/clustering.html)  
[Halakatti,Shashidhar T & Halakatti, Shambulinga T, 2017, IPASJ International Journal of Computer Science (IIJCS) Volume 5, Issue 8, "Identification Of Iris Flower Species Using
Machine Learning"](http://ipasj.org/IIJCS/Volume5Issue8/IIJCS-2017-08-18-18.pdf)



