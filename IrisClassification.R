# Iris Flower Classification
# Author : Vana Panagiotou
# Date : 03/10/2017

# Load iris data set
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"),
                   header = FALSE)

# Add meaningful column names
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

# coerce Species class from factor to character
iris$Species<-gsub("%", "", iris$Species)



#### EXPLORATORY ANALYSIS

# Find how many instances (rows) and how many attributes (columns) the data contains
dim(iris)

# list types for each attribute
sapply(iris, class)

# summarize the class distribution of the Species
percentage <- prop.table(table(iris$Species)) * 100
cbind(frequency = table(iris$Species), percentage = percentage)

# Print basic statistics
summary(iris)

# Analyze relationships between variables

#install.packages("GGally")
library(GGally)
# Make a matrix of plots
pairs <- ggpairs(iris, ggplot2::aes(colour=Species), title = "Iris Variables Relationships")
pairs
#install.packages("plotly")
library(plotly)
# create interactive graph
ggplotly(pairs) 

#install.packages("ggplot2")
library(ggplot2)
# Petal Length Vs.Petal Width
plot1 <- ggplot(iris, aes(x = Petal.Width, y = Petal.Length, shape = Species, color = Species)) +
       geom_point(aes(colour = Species, shape = Species)) +
       xlab("Petal Width") +  ylab("Petal Length") +
       ggtitle("Petal Length Vs. Petal Width") +
       annotate('text', x = 2,y = 2, label = paste("Correlation :", 
                  round(cor(iris$Petal.Length,iris$Petal.Width),3), sep=" "), size = 5)
plot1
# create interactive graph
ggplotly(plot1)

# Sepal Length Vs. Sepal Width
plot2 <- ggplot(iris, aes(x = Sepal.Width, y = Sepal.Length, shape = Species, color = Species)) +
       geom_point(aes(colour = Species, shape = Species)) +
       xlab("Sepal Width") +  ylab("Sepal Length") +
       ggtitle("Sepal Length Vs. Sepal Width") +
       annotate('text', x = 4,y = 6.5, label = paste("Correlation :", 
                round(cor(iris$Sepal.Length,iris$Sepal.Width),3), sep=" "), size = 5)
plot2
# create interactive graph
ggplotly(plot2)




##### CLASSIFICATION


# Create training/testing sets 

#install.packages("caret")
# Load the Caret package to partition the data
library(caret)
# Set the seed to ensure reproduceability
set.seed(1000)
# Create a partition (75% training, 25% testing)
index <- createDataPartition(iris$Species, p = 0.75, list = FALSE)
# select 75% of data to train the models
train_set <- iris[index,]
# select 25% of the data for testing
test_set <- iris[-index,]




# K-Means Clustering

# Set the seed to ensure reproduceability
set.seed(5)
# Since we know there are 3 classes, we begin with 3 centers. 
# Also since k-means assigns the centroids randomly we specify nstart as 20 to run 
# the algorithm 20 times with 20 random starting sets of centroids and then pick the best 
# of those 20
iris_Cluster <- kmeans(iris[, 1:4], centers = 3, nstart = 20)
iris_Cluster


# Check the classification accuracy
table(iris_Cluster$cluster, iris$Species)

# Plot the clusters and their centroids to see how the observations were clustered
# plot the Sepal attributes
plot(iris[c("Sepal.Length", "Sepal.Width")], col = iris_Cluster$cluster, 
     main = "Iris Classes with centroids based on Sepal attributes")
points(iris_Cluster$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3, pch=8, cex=2)

# plot the Petal attributes
plot(iris[c("Petal.Length", "Petal.Width")], col=iris_Cluster$cluster, 
     main ="Iris Classes with centroids based on Petal attributes")
points(iris_Cluster$centers[,c("Petal.Length", "Petal.Width")], col=1:3, pch=8, cex=2)

## convert cluster to categorical data
#iris_Cluster$cluster <- as.factor(iris_Cluster$cluster)




# Linear Discriminant Analysis

# Set the seed to ensure reproduceability
set.seed(5)
#install.packages("MASS")
library(MASS)
# Fit the model
model.lda <- train(x = train_set[,1:4], y = train_set[,5], method = "lda", metric = "Accuracy")
# Print the model
print(model.lda)

# Check how Linear Discriminant Analysis performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.lda, newdata = train_set[,1:4])
confusionMatrix(pred_labels,train_set$Species)


# Check how Linear Discriminant Analysis performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.lda, newdata = test_set[,1:4])
confusionMatrix(pred_labels_test, test_set$Species)



# Decision Tree Classifier

# Set the seed to ensure reproduceability
set.seed(5)
# Fit the model
model.rpart <- train(x = train_set[,1:4], y = train_set[,5], method = "rpart", metric = "Accuracy")
# Print the model
print(model.rpart)

# Plot the tree to see how the classification tree looks
#install.packages("rattle")
library(rattle)
# Plot decision tree
fancyRpartPlot(model.rpart$finalModel) 


# Check how Decision Tree performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.rpart,newdata = train_set[,1:4])
# Check the accuracy using a confusion matrix by comparing predictions to actual classifications
confusionMatrix(pred_labels,train_set$Species)


# Check how Decision Tree performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.rpart$finalModel, newdata = test_set[,1:4], type = "class")
confusionMatrix(pred_labels_test, test_set$Species)




# Random Forest Classifier

# Set the seed to ensure reproduceability
set.seed(5)
#install.packages("randomForest")
# Load Library Random FOrest
library(randomForest)
# Fit the model
model.rf <- train(x = train_set[,1:4], y = train_set[,5], method = "rf", metric = "Accuracy")
# Print the model
print(model.rf)

# Check how Random Forest performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.rf, newdata = train_set[,1:4])
# Check the accuracy using a confusion matrix by comparing predictions to actual classifications
confusionMatrix(pred_labels, train_set$Species)


# Check how Random Forest performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.rf$finalModel, newdata = test_set[,1:4], type = "class")
confusionMatrix(pred_labels_test, test_set$Species)




# Gradient Boosting Method

# Set the seed to ensure reproduceability
set.seed(5)
#install.packages("gbm")
# Load Library gbm
library(gbm)
# Fit the model
model.gbm <- train(x = train_set[,1:4], y = train_set[,5], method = "gbm", metric = "Accuracy",
                 verbose=FALSE)
# Print the model
print(model.gbm)


# Check how Gradient Boosting Method performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.gbm, newdata = train_set[,1:4])
confusionMatrix(pred_labels, train_set$Species)

# Check how Gradient Boosting Method performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.gbm$finalModel, n.trees=model.gbm$bestTune$n.trees, 
                   newdata = test_set[,1:4], type='response')

# Pick the response with the highest probability from the resulting pred_labels_test matrix, 
# by doing apply(.., 1, which.max) on the vector output from prediction
# in order to get categorical data from predict
pred_labels_test_categorical <- apply(pred_labels_test, 1, which.max)
pred_labels_class_labels <- colnames(pred_labels_test)[pred_labels_test_categorical]

confusionMatrix(pred_labels_class_labels ,test_set$Species)



# kNN (k-Nearest Neighbors)

# Set the seed to ensure reproduceability
set.seed(5)
# Fit the model
model.knn<-train(x = train_set[,1:4], y = train_set[,5], method = "knn", metric = "Accuracy")
# Print the model
print(model.knn)


# Check how kNN performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.knn, newdata = train_set[,1:4])
confusionMatrix(pred_labels, train_set$Species)

# Check how kNN performs on the testing set

## Performance on the testing set
pred_labels_test<-predict(object = model.knn$finalModel, newdata = test_set[,1:4], type = "class")
confusionMatrix(pred_labels_test, test_set$Species)




# Support Vector Machines (SVM) with Radial Basis Function Kernel 
#install.packages("kernlab")
# Load Library kernlab
library(kernlab)
# Set the seed to ensure reproduceability
set.seed(5)
# Fit the model
model.svm <- train(x = train_set[,1:4], y = train_set[,5], method = "svmRadial", metric = "Accuracy")
# Print the model
print(model.svm)

# Check how SVM performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.svm, newdata = train_set[,1:4])
confusionMatrix(pred_labels, train_set$Species)

# Check how SVM performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.svm$finalModel, newdata = test_set[,1:4], type="class")
confusionMatrix(pred_labels_test, test_set$Species)




# Neural Network 

# Set the seed to ensure reproduceability
set.seed(5)
# Fit the model
model.nnet<-train(x = train_set[,1:4], y = train_set[,5], method = "nnet", metric = "Accuracy", 
                  verbose = FALSE)
# Print the model
print(model.nnet)

# Check how Neural Network performs on the training set

## Performance on the training set
pred_labels <- predict(object = model.nnet, newdata = train_set[,1:4])
confusionMatrix(pred_labels, train_set$Species)

# Check how Neural Network performs on the testing set

## Performance on the testing set
pred_labels_test <- predict(object = model.nnet$finalModel, newdata = test_set[,1:4], type = "class")
confusionMatrix(pred_labels_test, test_set$Species)





######   Summarizing the Models

# summarize accuracy of models
results <- resamples(list(DecisionTree = model.rpart, RandomForest = model.rf, GBM = model.gbm,
                          LDA = model.lda, SVM = model.svm, NeuralNetwork = model.nnet, 
                          kNN = model.knn))
summary(results)


dotplot(results)