# A Prediction of Flower Species using the Iris Data Set in R

## Table of Contents
* [Project Description](#Project-Description)
* [Packages Required](#Packages-Required)
* [Descriptive Statistics and Exploratory Analysis](#Descriptive-Statistics-and-Exploratory-Analysis)
* [Splitting the Data into Training and Testing Sets](#Splitting-the-Data-into-Training-and-Testing-Sets)
* [Model Building](#Model-Building)


## <a name="Project-Description"></a> Project Description

This project is a classification task, whose purpose is to classify the iris flowers into their respective species based on available attributes. The three iris species include Setosa, Versicolor and Virginica, and the explanatory variables (attributes) include sepal length, sepal width, pedal length and petal width. 

## <a name="Packages-Required"></a> Packages Required

Here are the required packages to run the code properly:
```
GGally
plotly
ggplot2
caret
MASS
rattle
randomForest
gbm
kernlab
```

## <a name="Descriptive-Statistics-and-Exploratory-Analysis"></a> Descriptive Statistics and Exploratory Analysis 

First, we are going to take a look at the data. The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of the three species (Setosa, Versicolor or Virginica). Specifically, we can see that each class has the same number of instances (50 or 33% of the dataset).

We also use some visualizations (with ggplot, ggpairs and plotly) in order to better understand the relationships between the attributes.

## <a name="Splitting-the-Data-into-Training-and-Testing-Sets"></a> Splitting the Data into Training and Testing Sets

Before creating some models of the data, we need to split the dataset into training set and testing set. In this project, we used 75% of the data (chosen randomly) to build and train the models and the remaining 25% to assess how well the developed algorithms work. That is, we are going to use the training set in order to understand the data, select the appropriate model and determine the model parameters, and the testing set, which contains unseen data, in order to get a realistic and more concrete estimate of the models’ performance.

## <a name="Model-Building"></a> Model Building
Now, we will build and fit some models to the training set and we will estimate their accuracy on the testing set. For this purpose, we will build eight different models: K-Means Clustering, Linear Discriminant Analysis (LDA), Decision Tree (CART), Random Forest (RF), Gradient Boosting Method (GBM), k-Nearest Neighbor (kNN), Support Vector Machines (SVM) with Radial Basis Function Kernel and Neural Network (NN). Two of them are linear (LDA, K-Means Clustering), two are nonlinear (CART, kNN) and four are complex nonlinear (RF, GBM, SVM, NN). 

More details can be found within the project.

