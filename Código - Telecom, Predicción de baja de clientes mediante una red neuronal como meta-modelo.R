###########                       Preparación                        ########### 
# Librerias
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(mboost)) install.packages("mboost", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(pamr)) install.packages("pamr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

library(readr)
library(caret)
library(e1071)
library(ranger)
library(plyr)
library(dplyr)
library(mboost)
library(MASS)
library(klaR)
library(kernlab)
library(gbm)
library(pamr)
library(tidyverse)

# Data en Kaggle
# https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
churn.bigml.80 <- read.csv("../churn-bigml-80.csv")
churn.bigml.20 <- read.csv("../churn-bigml-20.csv")

dataset <- rbind(churn.bigml.80,churn.bigml.20)
rm(churn.bigml.80,churn.bigml.20)

any(is.na(dataset))
any(duplicated(dataset))
###########                  Pre - Procesamiento                     ########### 

dataset[sapply(dataset, is.character)] <- lapply(dataset[sapply(dataset, is.character)], 
                                       as.factor)
dataset$Churn <- factor(dataset$Churn, labels = c("No_churn", "Churn"))
dataset$Churn <- relevel(dataset$Churn, "Churn")
###########                  Creación del modelo                     ###########
set.seed(1, sample.kind = "Rounding") # solo para hacer el ejemplo reproducible

train_index <- createDataPartition(y = dataset$Churn, times = 1, p = 0.8, list = FALSE)

train_set <- dataset[train_index,]
test_set <- dataset[-train_index,]


folds <- createFolds(y = train_set$Churn, k = 10, list  = TRUE, returnTrain = FALSE)

#como quiero un dataframe de salida uso lapply

# Nota: en cada iteración entrenamos todos los modelos, que a sus vez usan CV,
# por lo que esto puede llevar mucho tiempo.
input_meta_model <- lapply(i <- 1:length(folds), function(i){
    
    print(i)
    
    train_folds <- train_set[unlist(folds[-i], use.names = FALSE),]
    test_fold <- train_set[folds[[i]],]
    
    ###### ENTRENAMIENTO ###

    #Random Forest
    print("ranger")
    fit_rf <- train(Churn ~ .,
                     data = train_folds,
                     method = "ranger",
                     trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                     preProcess = "nzv")
    #GLM
    print("glm")
    fit_glm <- train(Churn ~ .,
                     data = train_folds,
                     method = "glm",
                     trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                     preProcess = c("center", "scale", "nzv"))

    #LDA
    print("lda")
    fit_lda <- train(Churn ~ .,
                     data = train_folds,
                     method = "lda",
                     trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                     preProcess = c("center", "scale", "nzv"))
    
    #QDA
    print("qda")
    fit_qda <- train(Churn ~ .,
                          data = train_folds,
                          method = "qda",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"))
    
    #glmboost
    print("glmboost")
    fit_glmboost <- train(Churn ~ .,
                     data = train_folds,
                     method = "glmboost",
                     trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                     preProcess = c("center", "scale", "nzv"))
    
    #naive_bayes
    print("nb")
    fit_nb <- train(Churn ~ .,
                          data = train_folds,
                          method = "nb",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"))
    
    #svmLinear
    print("svmLinear")
    fit_svmLinear <- train(Churn ~ .,
                          data = train_folds,
                          method = "svmLinear",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"))
    
    #knn
    print("knn")
    fit_knn <- train(Churn ~ .,
                          data = train_folds,
                          method = "knn",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"))
    
    #svmRadialSigma
    print("svmRadialSigma")
    fit_svmRadialSigma <- train(Churn ~ .,
                          data = train_folds,
                          method = "svmRadialSigma",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"))
    
    #gbm
    print("gbm")
    fit_gbm <- train(Churn ~ .,
                          data = train_folds,
                          method = "gbm",
                          trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                          preProcess = c("center", "scale", "nzv"),
                          verbose = FALSE )
    
    #pam
    print("pam")
    fit_pam <- train(Churn ~ .,
                     data = train_folds,
                     method = "pam",
                     trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                     preProcess = c("center", "scale", "nzv"))
    
    ###### PREDICCIONES ###

    hat_rf <- predict(fit_rf, newdata = test_fold)
    hat_glm <- predict(fit_glm, newdata = test_fold)
    hat_lda <- predict(fit_lda, newdata = test_fold)
    hat_qda <- predict(fit_qda, newdata = test_fold)
    hat_glmboost <- predict(fit_glmboost, newdata = test_fold)
    hat_nb <- predict(fit_nb, newdata = test_fold)
    hat_svmLinear <- predict(fit_svmLinear, newdata = test_fold)
    hat_knn <- predict(fit_knn, newdata = test_fold)
    hat_svmRadialSigma <- predict(fit_svmRadialSigma, newdata = test_fold)
    hat_gbm <- predict(fit_gbm, newdata = test_fold)
    hat_pam <- predict(fit_pam, newdata = test_fold)
    
    ###### INPUT ###
    #guardo las predicciones(Y_hat) que seran los predictores(X) del meta modelo
    # y guardo la Y real que sera la que tratara de predecir el meta modelo
    
    print("------------------------------------------------")
    
    data.frame(
        hat_rf,
        hat_glm,
        hat_lda,
        hat_qda,
        hat_glmboost,
        hat_nb,
        hat_svmLinear,
        hat_knn,
        hat_svmRadialSigma,
        hat_gbm,
        hat_pam,
        
        Churn = test_fold$Churn)
})

input_meta_model <- do.call(rbind, input_meta_model)

nrow(input_meta_model)==nrow(train_set)

#### ML MODELS ###

#Random Forest
fit_rf <- train(Churn ~ .,
                data = train_set,
                method = "ranger",
                trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                preProcess = "nzv")
#GLM
fit_glm <- train(Churn ~ .,
                 data = train_set,
                 method = "glm",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"))

#LDA
fit_lda <- train(Churn ~ .,
                 data = train_set,
                 method = "lda",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"))

#QDA
fit_qda <- train(Churn ~ .,
                 data = train_set,
                 method = "qda",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"))

#glmboost
fit_glmboost <- train(Churn ~ .,
                      data = train_set,
                      method = "glmboost",
                      trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                      preProcess = c("center", "scale", "nzv"))

#naive_bayes
fit_nb <- train(Churn ~ .,
                data = train_set,
                method = "nb",
                trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                preProcess = c("center", "scale", "nzv"))

#svmLinear
fit_svmLinear <- train(Churn ~ .,
                       data = train_set,
                       method = "svmLinear",
                       trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                       preProcess = c("center", "scale", "nzv"))

#knn
fit_knn <- train(Churn ~ .,
                 data = train_set,
                 method = "knn",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"))

#svmRadialSigma
fit_svmRadialSigma <- train(Churn ~ .,
                            data = train_set,
                            method = "svmRadialSigma",
                            trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                            preProcess = c("center", "scale", "nzv"))

#gbm
fit_gbm <- train(Churn ~ .,
                 data = train_set,
                 method = "gbm",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"),
                 verbose = FALSE )

#pam
fit_pam <- train(Churn ~ .,
                 data = train_set,
                 method = "pam",
                 trControl = trainControl(method = "cv", number = 10, sampling = "up"),
                 preProcess = c("center", "scale", "nzv"))

hat_rf <- predict(fit_rf, newdata = test_set)
hat_glm <- predict(fit_glm, newdata = test_set)
hat_lda <- predict(fit_lda, newdata = test_set)
hat_qda <- predict(fit_qda, newdata = test_set)
hat_glmboost <- predict(fit_glmboost, newdata = test_set)
hat_nb <- predict(fit_nb, newdata = test_set)
hat_svmLinear <- predict(fit_svmLinear, newdata = test_set)
hat_knn <- predict(fit_knn, newdata = test_set)
hat_svmRadialSigma <- predict(fit_svmRadialSigma, newdata = test_set)
hat_gbm <- predict(fit_gbm, newdata = test_set)
hat_pam <- predict(fit_pam, newdata = test_set)


meta_model_test <- data.frame(
    hat_rf,
    hat_glm,
    hat_lda,
    hat_qda,
    hat_glmboost,
    hat_nb,
    hat_svmLinear,
    hat_knn,
    hat_svmRadialSigma,
    hat_gbm,
    hat_pam,
    
    Churn = test_set$Churn)

###########                       META MODEL                         ###########

fit_meta_model <- train(Churn ~ .,
                        data = input_meta_model,
                        method = "nnet",
                        tuneGrid = data.frame(size = seq(0,3),
                                              decay = seq(.0001,.1, length.out = 12)),
                        trControl = trainControl(method = "cv", number = 10, sampling = "up"))

hat_meta_model<- predict(fit_meta_model, newdata = meta_model_test)

fit_meta_model$bestTune
#test set y meta model test set tienen la misma Y, es lo mismo cual ponerlo acá
confusionMatrix(data = hat_meta_model, reference = test_set$Churn)



