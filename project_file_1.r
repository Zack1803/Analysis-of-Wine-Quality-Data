rm(list = ls())
library(ISLR)
library(ISLR2)

library(tree)
library(caret)

library(leaps)
library(faraway)

library(MASS)

library(GGally)
library(ggplot2)

library(randomForest)
library(pROC)
library(rpart)
library(plotly)
require(nnet)
library(RColorBrewer)
library(rattle)
library(RWeka)

#redwine data
redwine = read.csv("winequality-red.csv", header= T, na.strings="?", sep=";")
redwine = na.omit(redwine)
n_r = dim(redwine)
redwine_rows = nrow(redwine)
dim(redwine)
names(redwine)
table(redwine$quality)
str(redwine)
summary(redwine)

cor(redwine)

pairs(redwine)
dev.off()

#whitewine data
whitewine = read.csv("winequality-white.csv", header= T, na.strings="?", sep=";")
whitewine = na.omit(whitewine)
n_w = dim(whitewine)
whitewine_rows = nrow(whitewine)
head(redwine)
head(whitewine)
str(whitewine)
summary(whitewine)

cor(whitewine)

pairs(whitewine)
dev.off()

################################################################################

set.seed(1)

red_train.index = sample(redwine_rows, 0.7*redwine_rows)
redwine_train = redwine[red_train.index, ]
redwine_test = redwine[-red_train.index, ]

white_train.index = sample(whitewine_rows, 0.7*whitewine_rows)
whitewine_train = whitewine[white_train.index, ]
whitewine_test = whitewine[-white_train.index, ]


#multi linear regression for red wine
lm_red = lm(quality ~. , data = redwine_train)
summary(lm_red)

lm_red.predict = predict(lm_red, newdata = redwine_test, type="response")


lm_red_modified = lm(quality ~ volatile.acidity + residual.sugar + chlorides + 
                       total.sulfur.dioxide + sulphates + alcohol, data = redwine_train)
summary(lm_red_modified)
lm_red_modified.predict = predict(lm_red_modified, newdata = redwine_test, type="response")
mean((redwine_test$quality - lm_red_modified.predict)^2)
table(round(lm_red_modified.predict), redwine_test$quality)
(0+147+114+11)/redwine_rows
BIC(lm_red_modified)
AIC(lm_red_modified)

lm_red_modified.roc=roc(redwine_test$quality ~ lm_red_modified.predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(lm_red_modified.roc)
ggroc(lm_red_modified.roc)


#multi linear regression for white wine
lm_white = lm(quality ~. , data = whitewine_train)
summary(lm_white)
lm_white.predict = predict(lm_white, newdata = whitewine_test, type="response")

lm_white_modified = lm(quality ~ fixed.acidity + volatile.acidity + residual.sugar +free.sulfur.dioxide +
                         density + pH + sulphates + alcohol, data = whitewine_train)
summary(lm_white_modified)
lm_white_modified.predict = predict(lm_white_modified, newdata = whitewine_test, type="response")
mean((whitewine_test$quality - lm_white_modified.predict)^2)
table(round(lm_white_modified.predict), whitewine_test$quality)

BIC(lm_white_modified)
AIC(lm_white_modified)

lm_white_modified.roc=roc(whitewine_test$quality ~ lm_white_modified.predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(lm_white_modified.roc)
ggroc(lm_white_modified.roc)

# multinomial logistic regression for red wine
glm_red_modified = multinom(quality ~ volatile.acidity + residual.sugar + chlorides + 
                              total.sulfur.dioxide + sulphates + alcohol, data = redwine_train)
summary(glm_red_modified)
glm_red_modified.predict = predict(glm_red_modified, newdata = redwine_test)
confusionMatrix(as.factor(glm_red_modified.predict), as.factor(redwine_test$quality))
#length(glm_red_modified.predict)
BIC(glm_red_modified)
AIC(glm_red_modified)

glm_red_modified.roc=roc(redwine_test$quality ~ glm_red_modified.predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(glm_red_modified.roc)
ggroc(glm_red_modified.roc)

#multinomial logistic regression for white wine
glm_white_modified = multinom(quality ~ fixed.acidity + volatile.acidity + residual.sugar +free.sulfur.dioxide +
                                density + pH + sulphates + alcohol, data = whitewine_train)
summary(glm_white_modified)
glm_white_modified.predict = predict(glm_white_modified, whitewine_test)
confusionMatrix(as.factor(glm_white_modified.predict), as.factor(whitewine_test$quality))

BIC(glm_white_modified)
AIC(glm_white_modified)

glm_white_modified.roc=roc(whitewine_test$quality ~ glm_white_modified.predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(glm_white_modified.roc)
ggroc(glm_white_modified.roc)

#decision tree for redwine

redwine_tree = tree(quality~ volatile.acidity + residual.sugar + chlorides + 
                      total.sulfur.dioxide + sulphates + alcohol, data = redwine_train)
summary(redwine_tree)
plot(redwine_tree)
text(redwine_tree, pretty = 0)

prune = prune.tree(redwine_tree, best = 6 )
summary(prune)
plot(prune)
text(prune, pretty = 0)

redWine_predict = predict(prune, newdata = redwine_test)
#length(redWine_predict)
plot(redwine_test$quality, redWine_predict)
abline(a = 0, b = 1)
confusionMatrix(as.factor(round(redWine_predict)), as.factor(redwine_test$quality))
mean((redwine_test$quality - redWine_predict)^2)
table(round(redWine_predict),redwine_test$quality)
(127 + 106 + 24)/redwine_rows

prune.roc=roc(redwine_test$quality ~ redWine_predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(prune.roc)
ggroc(prune.roc)

#decision tree for whitewine

whitewine_tree = tree(quality ~ fixed.acidity + volatile.acidity + residual.sugar +free.sulfur.dioxide +
                        density + pH + sulphates + alcohol, data = whitewine_train)
summary(whitewine_tree)
plot(whitewine_tree)
text(whitewine_tree, pretty = 0)

prune.wd = prune.tree(whitewine_tree, best = 6 )
summary(prune.wd)
plot(prune.wd)
text(prune.wd, pretty = 0)

whitewine_predict = predict(prune.wd, newdata = whitewine_test)
plot(whitewine_test, whitewine_predict)
abline(a = 0, b = 1)
confusionMatrix(as.factor(round(whitewine_predict)), as.factor(whitewine_test$quality))
mean((whitewine_test$quality - whitewine_predict)^2)
table(round(whitewine_predict), whitewine_test$quality)
(280 + 364 + 98)/whitewine_rows

prune.roc=roc(whitewine_test$quality ~ whitewine_predict, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(prune.roc)
ggroc(prune.roc)

##random forest for red and white wines 

redwine_randomforest = randomForest(quality ~ volatile.acidity + residual.sugar + chlorides + 
                                      total.sulfur.dioxide + sulphates + alcohol, data = redwine_train)
summary(redwine_randomforest)
redwine_randomforest
prediction <- predict(redwine_randomforest, newdata = redwine_test)
confusionMatrix(as.factor(round(prediction)), as.factor(redwine_test$quality))
#mean((redwine$quality - prediction)^2)
table(round(prediction), redwine_test$quality)

prediction.roc=roc(redwine_test$quality ~ prediction, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(prediction.roc)
ggroc(prediction.roc)


whitewine_randomforest = randomForest(quality ~ fixed.acidity + volatile.acidity + residual.sugar +free.sulfur.dioxide +
                                        density + pH + sulphates + alcohol, data = whitewine_train)
summary(whitewine_randomforest)
whitewine_randomforest
predictions <- predict(whitewine_randomforest, newdata = whitewine_test)
confusionMatrix(as.factor(round(predictions)), as.factor(whitewine_test$quality))
#mean((whitewine$quality - predictions)^2)
table(round(predictions), whitewine_test$quality)

predictions.roc=roc(whitewine_test$quality ~ predictions, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(predictions.roc)
ggroc(predictions.roc)


`#regression tree r part

redwine.rpart <- rpart(quality ~ volatile.acidity + residual.sugar + chlorides + 
                         total.sulfur.dioxide + sulphates + alcohol, data = redwine_train)
redwine.rpart
fancyRpartPlot(redwine.rpart)

redwine.rpart.pred <- predict(redwine.rpart,redwine_test)
summary(redwine.rpart.pred)
confusionMatrix(as.factor(round(redwine.rpart.pred)), as.factor(redwine_test$quality))
mean((redwine_test$quality - redwine.rpart.pred)^2)
table(round(redwine.rpart.pred), redwine_test$quality)

redwine.rpart.pred.roc=roc(redwine_test$quality ~ redwine.rpart.pred, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(redwine.rpart.pred.roc)
ggroc(redwine.rpart.pred.roc)


whitewine.rpart <- rpart(quality ~fixed.acidity + volatile.acidity + residual.sugar +free.sulfur.dioxide +
                           density + pH + sulphates + alcohol, data = whitewine_train)
whitewine.rpart
fancyRpartPlot(whitewine.rpart)

whitewine.rpart.pred <- predict(whitewine.rpart,whitewine_test)
summary(whitewine.rpart.pred)
confusionMatrix(as.factor(round(whitewine.rpart.pred)), as.factor(whitewine_test$quality))
mean((whitewine_test$quality - whitewine.rpart.pred)^2)
table(round(whitewine.rpart.pred), whitewine_test$quality)

whitewine.rpart.pred.roc = roc(whitewine_test$quality ~ whitewine.rpart.pred, plot = TRUE, print.auc = TRUE)  #ROC curve
auc(whitewine.rpart.pred.roc)
ggroc(whitewine.rpart.pred.roc)

