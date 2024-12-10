library(dplyr)
library(caret)  
library(glmnet)
require(MASS)
library(leaps)
library(caret)
library(glmnet)
library(class)
library(randomForest)


#===============================================================================
# Exploratory Data Analysis and Data Prep ======================================

data_raw <- read.csv("~/school/fall2024/mis649/finalproject/NFL.csv", header = TRUE, na.string = "NA")

# Check for missing values
print(colSums(is.na(data_raw)))

# Replace missing values for numeric columns with the mean
numeric_cols <- sapply(data_raw, is.numeric)
data_raw[, numeric_cols] <- lapply(data_raw[, numeric_cols], function(x) {
  ifelse(is.na(x), mean(x, na.rm = TRUE), x)
})

# Replace missing values for categorical columns with "Unknown"
categorical_cols <- sapply(data_raw, is.factor) | sapply(data_raw, is.character)
data_raw[, categorical_cols] <- lapply(data_raw[, categorical_cols], function(x) {
  ifelse(is.na(x), "Unknown", x)
})

nfl <- data_raw

colnames(nfl) <- tolower(colnames(nfl))
# Verify no missing values
print(colSums(is.na(nfl)))

# Make sure position groups are accurate
nfl$position <- recode(nfl$position,
                       "C" = "DB",
                       "CB" = "DB",
                       "FB" = "RB",
                       "FS" = "S",
                       "SS" = "S",
                       "ILB" = "LB",
                       "OLB" = "LB",
                       "DE" = "DL",
                       "DT" = "DL",
                       "OG" = "OL",
                       "OT" = "OL",
                       "K" = "SP",
                       "P" = "SP",
                       "LS" = "SP")

# Make sure categorical variables are factors
nfl$drafted <- as.factor(nfl$drafted)
nfl$player_type <- as.factor(nfl$player_type)
nfl$position_type <- as.factor(nfl$position_type)
nfl$position <- as.factor(nfl$position)

# Examine variables and statistics
attach(nfl)
str(nfl)
summary(nfl)

# Pairs plot of relevant variables
pairs(nfl[,-c(1, 2, 3, 4, 13, 15, 16)])

# Regression: examine variable relationships
hist(sprint_40yd)
boxplot(sprint_40yd ~ position_type)
boxplot(sprint_40yd ~ position)
plot(weight,bmi)
plot(weight,sprint_40yd)
plot(weight*bmi,sprint_40yd)
plot(weight*weight,sprint_40yd)

# Classification: examine variable relationships
drafted_yes <- nfl[nfl$drafted == "Yes", ]

position_amount_yes <- table(drafted_yes$position)
barplot(position_amount_yes,
        main = "Number of Players Drafted by Position",
        xlab = "Position",
        ylab = "Number of Players Drafted")  

plot(drafted, sprint_40yd, 
     main = "Players Drafted vs 40 Yard Dash",
     xlab = "Drafted",
     ylab = "40yd Speed")
plot(drafted, bmi,
     main = "Players Drafted vs BMI",
     xlab = "Drafted",
     ylab = "BMI")

# Examine amounts of yes and no
drafted_amount <- table(drafted)
print(drafted_amount)
barplot(drafted_amount, 
        main = "Drafted Status Amounts", 
        xlab = "Drafted", 
        ylab = "Amount")


#===============================================================================
# Regression Setup
#===============================================================================

# Define model form
model_form <- (sprint_40yd ~ weight + I(weight^2) + weight:bmi + bmi 
               + vertical_jump + bench_press_reps + broad_jump + agility_3cone 
               + shuttle + position_type + position)

# Split data into train and test sets
n_tot <- nrow(nfl)                                  
pct_train <- .7                                          
n_train <- pct_train * n_tot                           

set.seed(123)                                           

train_ind <- sample(seq_len(n_tot), size = n_train)     
train_data <- nfl[train_ind, ]                               
test_data <- nfl[-train_ind, ]  

# Specify train and test variables for regression
x_train <- model.matrix(model_form, data = train_data)[, -1]
y_train <- train_data$sprint_40yd

x_test <- model.matrix(model_form, data = test_data)[, -1]
y_test <- test_data$sprint_40yd

# Define number of folds for k-fold cross validation
k = 10
kfolds <- createFolds(1:nrow(x_train), k = 10, list = TRUE, returnTrain = TRUE)

#===============================================================================
# Best subset selection with 10-fold cross validation 
#===============================================================================

# Function to extract each prediction model
subsets_pred <- function(model, dataframe, index) {
  mat <- model.matrix(model_form, data = dataframe)
  coefi <- coef(model, id = index)              
  mat[, names(coefi)] %*% coefi         
}

# Define model
model_form <- (sprint_40yd ~ weight + I(weight^2) + weight:bmi + bmi 
               + vertical_jump + bench_press_reps + broad_jump + agility_3cone 
               + shuttle + position_type + position)
best_subset <- regsubsets(model_form, data = nfl, nvmax = 20)

best_subset_summary <- summary(best_subset)
summary(best_subset)

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
cv_errors_bs <- matrix(NA, nrow = k, ncol = 20)  

for(j in 1:k) {
  # Train and test sets
  train <- nfl[kfolds[[j]], ]
  test <- nfl[-kfolds[[j]], ]
  
  # Fit model
  bsfit <- regsubsets(model_form, data = train, nvmax = 20)
  
  # Find test mse 
  for(i in 1:20) {
    pred_bs <- subsets_pred(bsfit, test, index = i)
    cv_errors_bs[j, i] <- mean((test$sprint_40yd - pred_bs)^2) 
  }
}

# Average the cv errors for all folds
cv_errors_bs_mean <- colMeans(cv_errors_bs)

# Plot cv errors for all subset models
plot(1:20, cv_errors_bs_mean, type = "o", pch = 19,
     xlab = "Number of Predictors", ylab = "CV MSE",
     main = "10-Fold CV MSE for Best Subset Selection")

summary(bsfit)

# Pick best model and report coefficients and test error
best_bs_model <- lm(sprint_40yd ~ + weight:bmi + bmi + vertical_jump
                    + bench_press_reps + broad_jump + agility_3cone 
                    + shuttle + position) 
summary(best_bs_model)

test_bs <- predict(best_bs_model, test_data)
test_error_bs <- mean((test_data$sprint_40yd - test_bs)^2)

# Residual analysis plots
par(mfrow=c(2,2))
plot(best_bs_model)
par(mfrow=c(1,1))


#===============================================================================
# Ridge regression with 10-fold cross validation
#===============================================================================

# Matrix for storing cross validation errors
cv_errors_ridge <- numeric(k)  

for (j in 1:k) {
  train_inds <- kfolds[[j]]
  train_x <- x_train[train_inds, ]
  train_y <- y_train[train_inds]
  test_x <- x_train[-train_inds, ]
  test_y <- y_train[-train_inds]
  
  # Fit ridge model
  ridge_model <- glmnet(train_x, train_y, alpha = 0, lambda = 10^seq(10, -2, length = 100), standardize = TRUE)
  
  # Make predictions
  cv_ridge <- cv.glmnet(train_x, train_y, alpha = 0, lambda = 10^seq(10, -2, length = 100), standardize = TRUE, foldid = rep(1:k, length.out = length(train_y)))
  ridge_best_lambda <- cv_ridge$lambda.min
  
  test_pred <- predict(ridge_model, s = ridge_best_lambda, newx = test_x)
  
  # Compute test error
  cv_errors_ridge[j] <- mean((test_y - test_pred)^2)
}

# Average cv error 
test_error_ridge <- mean(cv_errors_ridge)

# Examine nonzero predictor coefficients (beyond a threshold)
threshold <- 1e-4
ridge_coef <- as.matrix(coef(ridge_model, s = ridge_best_lambda))
print(ridge_coef)
non_zero_predictors_ridge <- rownames(ridge_coef)[abs(ridge_coef) > threshold]
non_zero_predictors_ridge <- non_zero_predictors_ridge[-1] 
print(non_zero_predictors_ridge)


# Plot coefficient paths
plot(ridge_model, xvar = "lambda", label = TRUE, xlim = c(-5, 10))  

# Plot lambda graph
plot(cv_ridge)


#===============================================================================
# Classification Setup
#===============================================================================

# Define model form
model_form1 <- (drafted ~ height + weight + bmi + sprint_40yd + bench_press_reps +
                  vertical_jump + broad_jump + agility_3cone + position)

# Create cv control
set.seed(123) 
train_control <- trainControl(method = "cv", number = 10) 

#===============================================================================
# Logistic Regression
#===============================================================================

# Training start time
start_time_logreg <- proc.time()

# Train logreg model
logreg_cv_model <- train(
  model_form1,
  data = nfl,
  method = "glm",  
  family = "binomial",
  trControl = train_control
)

# Model summary
print(logreg_cv_model)

# Print all coefficients
logreg_coefficients <- coef(logreg_cv_model$finalModel)
print(logreg_coefficients)

# Make predictions
logreg_preds <- predict(logreg_cv_model, nfl)

# Confusion matrix
conf_matrix_logreg <- confusionMatrix(logreg_preds, nfl$drafted)
print(conf_matrix_logreg)

# Compute training time
end_time_logreg <- proc.time()
logreg_training_time <- end_time_logreg - start_time_logreg
print(paste("Training time:", round(logreg_training_time["elapsed"], 5), "seconds"))

#===============================================================================
# Linear Discriminant Analysis
#===============================================================================

# Training start time
start_time_lda <- proc.time()

# Train lda model 
lda_cv_model <- train(
  model_form1,
  data = nfl,
  method = "lda",
  trControl = train_control
)

# Print the model summary
print(lda_cv_model)

# Make predictions on the dataset
lda_preds <- predict(lda_cv_model, nfl)

# Confusion matrix
conf_matrix_lda <- confusionMatrix(lda_preds, nfl$drafted)
print(conf_matrix_lda)


# Compute training time
end_time_lda <- proc.time()
lda_training_time <- end_time_lda - start_time_lda
print(paste("Training time:", round(lda_training_time["elapsed"], 5), "seconds"))


#===============================================================================
# KNN
#===============================================================================

# Define hyperparameter grid for k
k_values <- expand.grid(k = seq(1, 20, by = 2)) 

# Training start time
start_time_knn <- proc.time()

# Train KNN model with caret
knn_cv_model <- train(
  model_form1,
  data = nfl,
  method = "knn",
  tuneGrid = k_values,  
  trControl = train_control
)

# Print the best model and cv results
print(knn_cv_model)

# Make predictions on the dataset
knn_preds <- predict(knn_cv_model, nfl)

# Confusion matrix
conf_matrix_knn <- confusionMatrix(knn_preds, nfl$drafted)
print(conf_matrix_knn)

# Compute training time
end_time_knn <- proc.time()
knn_training_time <- end_time_knn - start_time_knn

# Print the optimal k value
optimal_k <- knn_cv_model$bestTune$k
print(paste("Optimal value for K:", optimal_k))


print(paste("Training time:", round(knn_training_time["elapsed"], 5), "seconds"))

#===============================================================================
# Random Forest
#===============================================================================

# Hyperparameter tuning with a grid search
tuning_grid <- expand.grid(.mtry = seq(2, sqrt(ncol(nfl) - 1), by = 1))  

# Training start time
start_time_rf <- proc.time()

# Train the rf model
rf_cv_model <- train(
  model_form1, 
  data = nfl,
  method = "rf",
  trControl = train_control,
  tuneGrid = tuning_grid,
  ntree = 100,
  importance = TRUE 
)

# Print best model and cv results
print(rf_cv_model)

# Make predictions using best model
rf_preds <- predict(rf_cv_model, nfl)

# Confusion matrix
conf_matrix <- confusionMatrix(rf_preds, nfl$drafted)
print(conf_matrix)

# Plot the importance of the features
varImpPlot(rf_cv_model$finalModel)

# Compute training time
end_time_rf <- proc.time()
rf_training_time <- end_time_rf - start_time_rf
print(paste("Training time:", round(rf_training_time["elapsed"], 5), "seconds"))