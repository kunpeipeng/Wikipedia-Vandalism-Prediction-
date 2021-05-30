# Wikipedia-Vandalism-Prediction

---
title: 'Individual Assignment 2: Wikipedia Vandalism'
author: "Kunpei Peng"
date: "2/6/2021"
output:
  word_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message=FALSE)
```

## Step 1: Exploratory Analysis
First, we can see that 1815 vandalism cases were detected in the history of this page. 
```{r}
#Load data and name it Wiki 
Wiki = read.csv("Wikipedia.csv")
str(Wiki)
View(Wiki)

#number of cases detected = 1815
Number_of_cases_detected <- sum(Wiki$Vandal)
```
The average number of words added is about 4.05 words, while the average number of words removed is about 3.5 words. 

```{r}
#finding the mean for column NumWordsAdded and NumWordsRemoved
Mean.word.added <- mean(Wiki$NumWordsAdded)
Mean.word.rmv <- mean(Wiki$NumWordsRemoved)

```

Let's explore the correlation between variables for a bit. From the correlation outputs below, we can see that variable "LoggedIn" is most correlated with variable Vandal as their correlation coefficient is about -0.43.  
```{r}
cor(Wiki)

```

## Step 2: Now let's randomly split the data into a training set and a testing set. In this case, I'm leaving 70% of the data in the training set. Let's see would would be the accuracy on the testing set of a simple baseline method that always predicts "not vandalism" for every edit.  

```{r}
#First, let's install all the useful packages for partioning data and creating classification models.
#install.packages("caTools")
library(caTools)
#install.packages("rpart")
library(rpart)
#install.packages("rpart.plot") 
library(rpart.plot)

#Before splitting the data, set seed so we can make sure that all our random partitions of our single dataset Wiki into training and testing data will be the same. (to replicate random results)

library(dplyr)
  
#Replace all N/A with 0 
Wiki %>% mutate_all(~replace(., is.na(.), 0))

#Convert data types to factor for validations later on. 

Wiki$Vandal <- as.factor(Wiki$Vandal)
Wiki$Minor <- as.factor(Wiki$Minor)
Wiki$LoggedIn <- as.factor(Wiki$LoggedIn)
Wiki$HTTP <- as.factor(Wiki$HTTP)
Wiki$NumWordsAdded <- as.factor(Wiki$NumWordsAdded)
Wiki$NumWordsRemoved <- as.factor(Wiki$NumWordsRemoved)

#set seed for reproducible splitting outcomes
set.seed(1234)

#Now split the data, randomly assigning 70% to training and 30% to testing. 
Split = sample.split(Wiki$Vandal, SplitRatio = 0.70)

Wiki.Train <- subset(Wiki, Split == TRUE)
Wiki.Test <- subset(Wiki, Split == FALSE)
```

Here, I Built a CART model as a baseline to predict Vandal, using all of the other variables as independent variables.

```{r}
#Now we need to tune the CART model using caret package. Here we used a 10 fold valiadation to find the best parameter for our data. 

library(caret)

#Here's how I decided on the number of folds for the validation: K = N/N*(%size of testing test) where N = Size of data set and K = Fold. In our case, we have 3876 rows of data and the testing set is 30%, so we can use about 3 folds forour CART1 cross validation. (3876/(0.3*3876))
#install.packages("MLmetrics")
library("MLmetrics")


Control=trainControl(method= "repeatedcv",number=3,repeats=3,classProbs=TRUE,summaryFunction=multiClassSummary)
cart_baseline=caret::train(make.names(Vandal)~.,data=Wiki.Train,method="rpart",trControl=Control,tuneLength=3)

cart_baseline

```
Note: Accuracy was used to select the optimal model using the largest value. As a result, 0.002755906 was selected as the optimal cp value to test the model. 


Plot the CART tree. Seems like the 'LoggedIn', 'NumWordsA', and 'NumWordsR' are selected by the model to be the top predictors.
```{r}
par(mar=c(1,1,1,1))
prp(cart_baseline$finalModel)
```

Check the accuracy of the model on the test set.
```{r}
#Predict the values of Vandal using the testing data set as input of our trained model WikiTree. Note that the default setting of the predict function below assumes a 0.5 threshold and use 0.5 as a probability benchmark to predict 0 or 1 scores. 

cart_baseline.Predicts <- predict(cart_baseline, newdata = Wiki.Test, type = "raw")


#After we got the predictions, let's compare the predicted values to the actual values. 
Comparison.Table <- table(Wiki.Test$Vandal, cart_baseline.Predicts)
Comparison.Table
WikiTree.Accuracy <- sum(diag(Comparison.Table))/sum(Comparison.Table)
WikiTree.Accuracy
```
In the confusion matrix above, we can see that the model accuracy is $0.7128117$, meaning our model accurately predicted about 71.28% of all vandalism on the page. 


## Step 3: Now we have a basline CART model, let's build a more sophisticated random forest model to predict "Vandal", using all the other variables as independent variables. 
```{r}
#install.packages("randomForest")
library(randomForest)
#install.packages("caret")
library(caret)

#Convert the independent variables(Training & Testing) from factors back to integers
Wiki.Train$Minor <- as.integer(Wiki.Train$Minor)
Wiki.Train$LoggedIn <- as.integer(Wiki.Train$LoggedIn)
Wiki.Train$HTTP <- as.integer(Wiki.Train$HTTP)
Wiki.Train$NumWordsAdded <- as.integer(Wiki.Train$NumWordsAdded)
Wiki.Train$NumWordsRemoved <- as.integer(Wiki.Train$NumWordsRemoved)

Wiki.Test$Minor <- as.integer(Wiki.Test$Minor)
Wiki.Test$LoggedIn <- as.integer(Wiki.Test$LoggedIn)
Wiki.Test$HTTP <- as.integer(Wiki.Test$HTTP)
Wiki.Test$NumWordsAdded <- as.integer(Wiki.Test$NumWordsAdded)
Wiki.Test$NumWordsRemoved <- as.integer(Wiki.Test$NumWordsRemoved)

#Find the right number of mtry
Wiki.RF.values <- vector(length=5)
for(i in 1:5) {
  Wiki.RF <- randomForest(Vandal ~ ., data = Wiki.Train, mtry = i, ntree = 500)
  Wiki.RF.values[i] <- Wiki.RF$err.rate[nrow(Wiki.RF$err.rate),1]
}

Wiki.RF.values
# As we can see from above, the 3rd value correspond to metry = 3has the lowest out of bag error rate. So we can just got with mtry = 3 in this model. 

Wiki.RF <- randomForest(Vandal ~., data = Wiki.Train, ntree=500, mtry = 3, proximity = TRUE)
Wiki.RF

Wiki.RF.Predict <- predict(Wiki.RF, newdata = Wiki.Test, type = "class")

#After we got the predictions, let's compare the predicted values to the actual values. 
RF.Comparison.Table <- table(Wiki.Test$Vandal, Wiki.RF.Predict)
Comparison.Table
Wiki.RF.Accuracy <- sum(diag(RF.Comparison.Table))/sum(RF.Comparison.Table)
Wiki.RF.Accuracy


```
Our random forest model accuracy turns out to be 73.68%, which is slightly better than our CART model(71.28% accuracy) by 2.4%. 

## Business Implications: 
Both of the models I built would be useful as baseline models for Wikipedia to detect vandalism. The model accuracy can be improved with the addition of more sophisticated techniques to optimize the model robustness. However, as the baseline models, the two models I built above should suffice as bench marks for Wikipedia. 

If I could collect more data about the edits, I would want to use variables such as 1)the associated editor's edit reversal records, and 2)the number of historical edits happened to the page. 

I wanted the variables mentioned above because 1) if the associated editor's edits to the page has been reversed multiple times, that associated editors is likely to be a habitual vandals. This can be a strong indicator of whether the edits would likely to be vandalism or not. 2) The number of historical edits for the page could be an indicator as well because if there's a high number of historical edits, there might be higher number of vandalism happen to the page because it's well known. 












