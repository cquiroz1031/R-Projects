require(MASS)
library(leaps)
library(caret)
library(glmnet)
library(pls)

auto_raw <- read.csv("https://www.statlearning.com/s/Auto.csv", header=T, na.string="?") 

auto <- na.omit(auto_raw)

attach(auto)

cylinders <- as.factor(cylinders)
origin <- as.factor(origin)


# (1) Split data into training and test sets
n_tot <- nrow(auto)                                  
pct_train <- .7                                          
n_train <- pct_train * n_tot                           

set.seed(123)                                           

train_ind <- sample(seq_len(n_tot), size = n_train)     
train_data <- auto[train_ind, ]                               
test_data <- auto[-train_ind, ]  


# (2) Useful predictor terms
pairs(auto[,-9])

boxplot(mpg ~ cylinders)
boxplot(mpg ~ origin)
plot(log(displacement),mpg)
plot(log(horsepower),mpg)
plot(log(weight),mpg)

#================================================================================================================
#================================================================================================================

# (3) Best subset selection and 10-fold cross validation 
#================================================================================================================

# Function to extract each prediction model
subsets_pred <- function(model, dataframe, index) {
  mat <- model.matrix(model_form, data = dataframe)
  coefi <- coef(model, id = index)              
  mat[, names(coefi)] %*% coefi         
}


model_form <- mpg ~ cylinders + origin + year + acceleration + log(displacement) + log(horsepower) + log(weight) + weight:horsepower + horsepower:displacement
best_subset <- regsubsets(model_form, data = auto, nvmax = 9)

best_subset_summary <- summary(best_subset)

# Kfold cv
k = 10
kfolds <- createFolds(mpg, k = 10, list = TRUE, returnTrain = TRUE)

# Plot bic
plot(best_subset_summary$bic, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "BIC",
     main = "BIC for Best Subset Selection")

# Plot adjusted r2
plot(best_subset_summary$adjr2, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Adjusted R-squared",
     main = "Adjusted R-squared for Best Subset Selection")

# Plot cp
plot(best_subset_summary$cp, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Cp",
     main = "Cp for Best Subset Selection")

# Matrix for storing cross validation errors
cv_errors_bs <- matrix(NA, nrow = k, ncol = 9)  

for(j in 1:k) {
  # Train and test sets
  train <- auto[kfolds[[j]], ]
  test <- auto[-kfolds[[j]], ]
  
  # Fit model
  bsfit <- regsubsets(model_form, data = train, nvmax = 9)
  
  # Find test mse 
  for(i in 1:9) {
    pred_bs <- subsets_pred(bsfit, test, index = i)
    cv_errors_bs[j, i] <- mean((test$mpg - pred_bs)^2) 
  }
}

# Average the cv errors for all folds
cv_errors_bs_mean <- colMeans(cv_errors_bs)

# Plot cv errors for all subset models
plot(1:9, cv_errors_bs_mean, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "CV MSE",
     main = "10-Fold CV MSE for Best Subset Selection")

summary(bsfit)

# Pick best model and report coefficients and test error.
best_bs_model <- lm(mpg ~ origin + year + acceleration + log(displacement) + log(horsepower) + log(weight) + horsepower:displacement, data=train_data)
summary(best_bs_model)

test_bs <- predict(best_bs_model, test_data)
test_error_bs <- mean((test_data$mpg - test_bs)^2)

#================================================================================================================
#================================================================================================================

# (4) Forward stepwise selection 
#===========================================================================================================

forward_subset <- regsubsets(model_form, data = auto, nvmax = 9, method="forward")

forward_subset_summary <- summary(forward_subset)


# Plot bic
plot(forward_subset_summary$bic, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "BIC",
     main = "BIC for Forward Stepwise Selection")

# Plot adjusted r2
plot(forward_subset_summary$adjr2, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Adjusted R-squared",
     main = "Adjusted R-squared for Forward Stepwise Selection")

# Plot cp
plot(forward_subset_summary$cp, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Cp",
     main = "Cp for Forward Stepwise Selection")

# Matrix for storing cv errors
cv_errors_for <- matrix(NA, nrow = k, ncol = 9)  

for(j in 1:k) {
  # Train and test sets
  train <- auto[kfolds[[j]], ]
  test <- auto[-kfolds[[j]], ]
  
  # Fit model
  forfit <- regsubsets(model_form, data = train, nvmax = 9, method="forward")
  
  # Find test mse 
  for(i in 1:9) {
    pred_for <- subsets_pred(forfit, test, index = i)
    cv_errors_for[j, i] <- mean((test$mpg - pred_for)^2) 
  }
}

# Average the cv errors for all folds
cv_errors_for_mean <- colMeans(cv_errors_for)

# Plot cv errors for all subset models
plot(1:9, cv_errors_for_mean, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "CV MSE",
     main = "10-Fold CV for Forward Stepwise Selection")

summary(forfit)

# Pick best model and report coefficients and test error.
best_for_model <- lm(mpg ~ origin + year + acceleration + log(horsepower) + log(weight) + weight:horsepower, data=train_data)
summary(best_for_model)

test_for <- predict(best_for_model, test_data)
test_error_for <- mean((test_data$mpg - test_for)^2)

#================================================================================================================
#================================================================================================================

# (5) Backward stepwise selection 
#===========================================================================================================

backward_subset <- regsubsets(model_form, data = auto, nvmax = 9, method="backward")

backward_subset_summary <- summary(backward_subset)


# Plot bic
plot(backward_subset_summary$bic, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "BIC",
     main = "BIC for Backward Stepwise Selection")

# Plot adjusted r2
plot(backward_subset_summary$adjr2, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Adjusted R-squared",
     main = "Adjusted R-squared for Backward Stepwise Selection")

# Plot cp
plot(backward_subset_summary$cp, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "Cp",
     main = "Cp for Backward Stepwise Selection")

# Matrix for storing cv errors
cv_errors_back <- matrix(NA, nrow = k, ncol = 9)  

for(j in 1:k) {
  # Train and test sets
  train <- auto[kfolds[[j]], ]
  test <- auto[-kfolds[[j]], ]
  
  # Fit model
  backfit <- regsubsets(model_form, data = train, nvmax = 9, method="backward")
  
  # Find test error 
  for(i in 1:9) {
    pred_back <- subsets_pred(backfit, test, index = i)
    cv_errors_back[j, i] <- mean((test$mpg - pred_back)^2) 
  }
}

# Average the cv errors for all folds
cv_errors_back_mean <- colMeans(cv_errors_back)

# Plot cv errors for all subset models
plot(1:9, cv_errors_back_mean, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "CV MSE",
     main = "10-Fold CV for Backward Stepwise Selection")

summary(backfit)

# Pick best model and report coefficients and test error.
best_back_model <- lm(mpg ~ origin + year + acceleration + log(displacement) + log(horsepower) + log(weight) + horsepower:displacement, data=train_data)
summary(best_back_model)

test_back <- predict(best_back_model, test_data)
test_error_back <- mean((test_data$mpg - test_back)^2)

#================================================================================================================
#================================================================================================================

# (6) Ridge regression
#==============================================================================================================

# Specify train and test variables for the rest of the methods
x_train <- model.matrix(model_form, data = train_data)[, -1]
y_train <- train_data$mpg

x_test <- model.matrix(model_form, data = test_data)[, -1]
y_test <- test_data$mpg

# Cv ridge with a specified range of lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, lambda = 10^seq(10, -2, length = 100), standardize = TRUE)  

# Extract lambda which minimizes cv error
ridge_best_lambda <- cv_ridge$lambda.min

# Plot the cv results
plot(cv_ridge)
title("CV for Ridge Regression (lambda)")

# Plot coefficient paths
ridge_general_model <- glmnet(x_train, y_train, alpha=0)
plot(ridge_general_model, xvar = "lambda", label = TRUE)
title("Ridge Coefficient Paths")

# Use best lambda 
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = ridge_best_lambda, standardize = TRUE)

# Make ridge predictions
test_ridge_pred <- predict(ridge_model, s = ridge_best_lambda, newx = x_test)

# Find test error
test_error_ridge <- mean((y_test - test_ridge_pred)^2)

# Examine nonzero predictor coefficients
ridge_coef <- as.matrix(coef(ridge_model, s = ridge_best_lambda))
print(ridge_coef)
non_zero_predictors_ridge <- rownames(ridge_coef)[ridge_coef != 0]
non_zero_predictors_ridge <- non_zero_predictors_ridge[-1] 
print(non_zero_predictors_ridge)

#================================================================================================================
#================================================================================================================

# (7) Lasso regression 
#==============================================================================================================

# Cv lasso with a specified range of lambda
cv_lasso <- cv.glmnet(x_train, y_train, lambda = 10^seq(10, -2, length = 100), standardize = TRUE)  

# Extract lambda which minimizes cv error
lasso_best_lambda <- cv_lasso$lambda.min

# Plot the cv results
plot(cv_lasso)
title("CV for Lasso Regression (lambda)")

# Plot coefficient paths
lasso_general_model <- glmnet(x_train, y_train)
plot(lasso_general_model, xvar = "lambda", label = TRUE)
title("Lasso Coefficient Paths")

# Use best lambda 
lasso_model <- glmnet(x_train, y_train, lambda = lasso_best_lambda, standardize = TRUE)

# Make ridge predictions
test_lasso_pred <- predict(lasso_model, s = lasso_best_lambda, newx = x_test)

# Find test error
test_error_lasso <- mean((y_test - test_lasso_pred)^2)

# Examine nonzero predictor coefficients
lasso_coef <- as.matrix(coef(lasso_model, s = lasso_best_lambda))
print(lasso_coef)
non_zero_predictors_lasso <- rownames(lasso_coef)[lasso_coef != 0]
non_zero_predictors_lasso <- non_zero_predictors_lasso[-1] 
print(non_zero_predictors_lasso)

#================================================================================================================
#================================================================================================================

# (8) Principal component regression 
#==============================================================================================================

pcr_model <- pcr(y_train ~ x_train, scale = TRUE, validation = "CV", segments=10)
summary(pcr_model)

# Plot of cv error to visualize best M
validationplot(pcr_model, val.type = "MSEP")

# Find M which minimizes cv error 
M_pcr <- which.min(pcr_model$validation$PRESS)

# Make predictions
pcr_pred <- predict(pcr_model, x_test, ncomp = M_pcr)
test_error_pcr <- mean((y_test - pcr_pred)^2)

#================================================================================================================
#================================================================================================================

# (9) Partial least squares 
#==============================================================================================================

pls_model <- plsr(y_train ~ x_train, scale = TRUE, validation = "CV", segments = 10)
summary(pls_model)

# Plot of cv error to visualize best M
validationplot(pls_model, val.type = "MSEP")

# Find M which minimizes cv error 
M_pls <- which.min(pls_model$validation$PRESS)  

# Make predictions
pls_pred <- predict(pls_model, x_test, ncomp = M_pls)
test_error_pls <- mean((y_test - pls_pred)^2)

#================================================================================================================
#================================================================================================================

# (10) Compare results
#==============================================================================================================

# Variance in the dv mpg
mpg_variance <- var(test_data$mpg)

# Collect all mse values
final_results <- data.frame(
  Method = c("Best Subset", "FS", "BS", "Ridge", "Lasso", "PCR", "PLS"),
  MSE = c(test_error_bs, test_error_for, test_error_back, test_error_ridge, 
               test_error_lasso, test_error_pcr, test_error_pls)
)

# Mse ratio
final_results$MSE_Ratio <- final_results$MSE / mpg_variance

print(final_results)

# Percent diff between models
best_mse <- min(final_results$MSE)
final_results$percent_diff <- ((final_results$MSE - best_mse) / best_mse) * 100

# Plot the test mse for each method
barplot(final_results$MSE, names.arg = final_results$Method, 
        col = "skyblue", main = "Test MSE",
        xlab = "Method", ylab = "Test MSE", las = 2, cex.names = 0.8)

# Plot the percent difference
barplot(final_results$percent_diff, names.arg = final_results$Method,
        col = "skyblue", main = "Percent Difference from Lowest Test MSE",
        xlab = "Regression Method", ylab = "Percent Difference (%)", las = 2, cex.names = 0.8)



