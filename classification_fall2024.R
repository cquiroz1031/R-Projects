# ISL 4.8 Problem 14

require(MASS)

auto_raw = read.csv("https://www.statlearning.com/s/Auto.csv", header=T, na.string="?") 

auto = na.omit(auto_raw)

attach(auto)


# (a) Create binary variable, and add it to data frame

auto$mpg01 <- ifelse(mpg > median(mpg), 1, 0)

mpg01 <- as.factor(mpg01)


# (b) Explore data graphically to investigate association between mpg01 and other vars

pairs(auto[,-9])

par(mfrow=c(2,2))
boxplot(acceleration ~ mpg01)
boxplot(weight ~ mpg01)       
boxplot(horsepower ~ mpg01)   
boxplot(displacement ~ mpg01) 


# (c) Split the data into a training set and a test set.

n_tot = nrow(auto)                                  

pct_train = .7                                          

n_train = pct_train * n_tot                           

set.seed(123)                                           

train_ind = sample(seq_len(n_tot), size = n_train)     

train = auto[train_ind, ]                               

test = auto[-train_ind, ]                               


# -------------------------------------------------------------------------------------------

# (d)  LDA:  

lda_model <- lda(mpg01 ~ acceleration + weight + horsepower + displacement, data = train) 
lda_model

lda_pred <- predict(lda_model, test)
lda_predclass = lda_pred$class

confusionLDA <- table(lda_predclass, test$mpg01)
TP_lda <- confusionLDA[2,2]
FP_lda <- confusionLDA[2,1]
TN_lda <- confusionLDA[1,1]
FN_lda <- confusionLDA[1,2]

# overall accuracy
overall_accLDA <- mean(lda_predclass == test$mpg01)
# sensitivity
sens_lda <- TP_lda / (TP_lda + FN_lda)
# specificity
spec_lda <- TN_lda / (TN_lda + FP_lda)
# test error
test_errorLDA <- mean(lda_predclass != test$mpg01)

table(lda_predclass, test$mpg01)

# -------------------------------------------------------------------------------------------

# (e)  QDA:  

qda_model <- qda(mpg01 ~ acceleration + weight + horsepower + displacement, data = train)
qda_model

qda_predclass <- predict(qda_model, test)$class

confusionQDA <- table(qda_predclass, test$mpg01)
TP_qda <- confusionQDA[2,2]
FP_qda <- confusionQDA[2,1]
TN_qda <- confusionQDA[1,1]
FN_qda <- confusionQDA[1,2]

# overall accuracy
overall_accQDA <- mean(qda_predclass == test$mpg01)
# sensitivity
sens_qda <- TP_qda / (TP_qda + FN_qda)
# specificity
spec_qda <- TN_qda / (TN_qda + FP_qda)
# test error
test_errorQDA <- mean(qda_predclass != test$mpg01)

table(qda_predclass, test$mpg01)

# -------------------------------------------------------------------------------------------

# (f)  Logistic Regression: 

glm_model <- glm(mpg01 ~ acceleration + weight + horsepower + displacement, data = train, family = binomial)
summary(glm_model)

glm_probs <- predict(glm_model, newdata= test, type = "response")

glm_predclass <- ifelse(glm_probs > 0.5, 1, 0)  

confusionGLM <- table(glm_predclass, test$mpg01)
TP_glm <- confusionGLM[2,2]
FP_glm <- confusionGLM[2,1]
TN_glm <- confusionGLM[1,1]
FN_glm <- confusionGLM[1,2]

# overall accuracy
overall_accGLM <- mean(glm_predclass == test$mpg01) 
# sensitivity
sens_glm <- TP_glm / (TP_glm + FN_glm)
# specificity
spec_glm <- TN_glm / (TN_glm + FP_glm)
# test error
test_errorGLM <- mean(glm_predclass != test$mpg01)  

table(glm_predclass, test$mpg01)

# -------------------------------------------------------------------------------------------

# (g)  Naive Bayes: 

library(e1071)
nb_model <- naiveBayes(mpg01 ~ acceleration + weight + horsepower + displacement, data = train)
nb_model

nb_predclass <- predict(nb_model, test)

confusionNB <- table(nb_predclass, test$mpg01)
TP_nb <- confusionNB[2,2]
FP_nb <- confusionNB[2,1]
TN_nb <- confusionNB[1,1]
FN_nb <- confusionNB[1,2]

# overall accuracy
overall_accNB <- mean(nb_predclass == test$mpg01) 
# sensitivity
sens_nb <- TP_nb / (TP_nb + FN_nb)
# specificity
spec_nb <- TN_nb / (TN_nb + FP_nb)
# test error
test_errorNB <- mean(nb_predclass != test$mpg01)  

table(nb_predclass, test$mpg01)

# -------------------------------------------------------------------------------------------

# (g)  KNN: 

library(class)

ind_vars <- cbind(acceleration, weight, horsepower, displacement)

# Prepare independent variables
ind_vars_train <- cbind(train$acceleration, train$weight, train$horsepower, train$displacement)
ind_vars_test <- cbind(test$acceleration, test$weight, test$horsepower, test$displacement)


# K = 3
knn_predclass3 <- knn(ind_vars_train, ind_vars_test, train$mpg01, k=3)

confusionKNN3 <- table(knn_predclass3, test$mpg01)
TP_knn3 <- confusionKNN3[2,2]
FP_knn3 <- confusionKNN3[2,1]
TN_knn3 <- confusionKNN3[1,1]
FN_knn3 <- confusionKNN3[1,2]

# overall accuracy
overall_accKNN3 <- mean(knn_predclass3 == test$mpg01) 
# sensitivity
sens_knn3 <- TP_knn3 / (TP_knn3 + FN_knn3)
# specificity
spec_knn3 <- TN_knn3 / (TN_knn3 + FP_knn3)
# test error
test_errorKNN3 <- mean(knn_predclass3 != test$mpg01) 

table(knn_predclass3, test$mpg01)


# K = 7
knn_predclass7 <- knn(ind_vars_train, ind_vars_test, train$mpg01, k=7)

confusionKNN7 <- table(knn_predclass7, test$mpg01)
TP_knn7 <- confusionKNN7[2,2]
FP_knn7 <- confusionKNN7[2,1]
TN_knn7 <- confusionKNN7[1,1]
FN_knn7 <- confusionKNN7[1,2]

# overall accuracy
overall_accKNN7 <- mean(knn_predclass7 == test$mpg01) 
# sensitivity
sens_knn7 <- TP_knn7 / (TP_knn7 + FN_knn7)
# specificity
spec_knn7 <- TN_knn7 / (TN_knn7 + FP_knn7)
# test error
test_errorKNN7 <- mean(knn_predclass7 != test$mpg01) 

table(knn_predclass7, test$mpg01)


# K = 9
knn_predclass9 <- knn(ind_vars_train, ind_vars_test, train$mpg01, k=9)

confusionKNN9 <- table(knn_predclass9, test$mpg01)
TP_knn9 <- confusionKNN9[2,2]
FP_knn9 <- confusionKNN9[2,1]
TN_knn9 <- confusionKNN9[1,1]
FN_knn9 <- confusionKNN9[1,2]

# overall accuracy
overall_accKNN9 <- mean(knn_predclass9 == test$mpg01) 
# sensitivity
sens_knn9 <- TP_knn9 / (TP_knn9 + FN_knn9)
# specificity
spec_knn9 <- TN_knn9 / (TN_knn9 + FP_knn9)
# test error
test_errorKNN9 <- mean(knn_predclass9 != test$mpg01) 

table(knn_predclass9, test$mpg01)


# K = 11
knn_predclass11 <- knn(ind_vars_train, ind_vars_test, train$mpg01, k=11)

confusionKNN11 <- table(knn_predclass11, test$mpg01)
TP_knn11 <- confusionKNN11[2,2]
FP_knn11 <- confusionKNN11[2,1]
TN_knn11 <- confusionKNN11[1,1]
FN_knn11 <- confusionKNN11[1,2]

# overall accuracy
overall_accKNN11 <- mean(knn_predclass11 == test$mpg01) 
# sensitivity
sens_knn11 <- TP_knn11 / (TP_knn11 + FN_knn11)
# specificity
spec_knn11 <- TN_knn11 / (TN_knn11 + FP_knn11)
# test error
test_errorKNN11 <- mean(knn_predclass11 != test$mpg01) 

table(knn_predclass11, test$mpg01)


# K = 13
knn_predclass13 <- knn(ind_vars_train, ind_vars_test, train$mpg01, k=13)

confusionKNN13 <- table(knn_predclass13, test$mpg01)
TP_knn13 <- confusionKNN13[2,2]
FP_knn13 <- confusionKNN13[2,1]
TN_knn13 <- confusionKNN13[1,1]
FN_knn13 <- confusionKNN13[1,2]

# overall accuracy
overall_accKNN13 <- mean(knn_predclass13 == test$mpg01) 
# sensitivity
sens_knn13 <- TP_knn13 / (TP_knn13 + FN_knn13)
# specificity
spec_knn13 <- TN_knn13 / (TN_knn13 + FP_knn13)
# test error
test_errorKNN13 <- mean(knn_predclass13 != test$mpg01) 

table(knn_predclass13, test$mpg01)








