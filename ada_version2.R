cat("\014")
rm(list = ls())
library(ISLR2)
library(caret)
library(tree)
library(car)
library(visdat)
library(ggplot2)
set.seed(4444)
data("Carseats")

#PART 4 : EDA - Data PreProcessing
dim(Carseats) #400 data 11 variable
head(Carseats) 
summary(Carseats)
hist(Carseats$Sales)

sum(is.na(Carseats)) #No Null data
vis_dat(Carseats)
vis_miss(Carseats)
df  = data.frame(Carseats)
#4.1-Range Scaling
summary(df)
hist(df$Sales)
mean(df$Sales) #0.46 is mean of Sales.
df$Sales = as.factor(ifelse(df$Sales >= 7.5,"Good","Bad"))

ans_no <- length(df$Sales[df$Sales == "Bad"])
ans_yes <- length(df$Sales[df$Sales == "Good"])
ans_no / (ans_no + ans_yes) ##Balanced Dataset No Problem

#4.2-Splitting Data Train and Test

smp_size <- floor(0.8 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_ind, ]
test <- df[-train_ind, ]
test_sales = df$Sales[-train_ind]

#Part 5 : Logistic Model Fit and Numerical Results
# gmodel.1 = glm(Sales ~ ., data = df, family = binomial(link  = "logit"))
# summary(gmodel.1)
# vif(gmodel.1)

#Standarizad model to see which variable has most impact
process = preProcess(as.data.frame(df),method=c("range"))
df_std = predict(process, as.data.frame(df))
gmodel.std = glm(Sales ~ ., data = df_std, family = binomial(link  = "logit"))
summary(gmodel.std)

#5.1 Cross Validated Logistic Model with LOOCV method
data_ctrl <- trainControl(method = "LOOCV")
model_caret <- train(Sales ~ . ,data = df,trControl = data_ctrl,method = "glm",na.action = na.pass)
print(model_caret)

#Decision Tree -------------

tree_model_1 = tree(Sales ~ .,train)
plot(tree_model_1)
text(tree_model_1, pretty = 0)


#How model using test data
tree_pred = predict(tree_model_1, test, type = "class")
cfmatrix1=with(test,table(tree_pred,Sales));cfmatrix1
mean(1-(tree_pred != test_sales)) #%31lerde missclass error (?ok fazla)
fourfoldplot(cfmatrix1, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

#Pruning the tree and cross validation for help
cv_tree = cv.tree(tree_model_1, FUN = prune.misclass)
names(cv_tree)
plot(cv_tree$size,cv_tree$dev, type = "b")

#Pruned tree
bestprune <- cv_tree$size[which.min(cv_tree$dev)]
pruned_tree = prune.misclass(tree_model_1, best = bestprune)
plot(pruned_tree)
text(pruned_tree, pretty = 0)

#Test pruned tree
tree_pred_2 = predict(pruned_tree, test,type = "class")
cfmatrix2= with(test,table(tree_pred_2,Sales));cfmatrix2
mean(1-(tree_pred_2 != test_sales))

fourfoldplot(cfmatrix2, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

