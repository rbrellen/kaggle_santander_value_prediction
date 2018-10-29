# Load packages
library(tidyverse)
library(glmnet)

# Read in data
train <- readr::read_csv('kaggle_data/train.csv.zip')
test  <- readr::read_csv('kaggle_data/test.csv.zip')

# Identify median values of training set
medians <- train %>%
  dplyr::select_if(is.numeric) %>%
  dplyr::summarize_all(funs(median), na.rm = TRUE)

# Specify features and response variable
x_train <- train %>% 
  dplyr::select_if(is.numeric) %>%
  tidyr::replace_na(medians) %>%
  dplyr::select(-target) %>%
  as.matrix()

y_train <- train %>% 
  dplyr::select(target) %>%
  as.matrix()

x_test <- test %>% 
  dplyr::select_if(is.numeric) %>%
  tidyr::replace_na(medians) %>%
  as.matrix()

# Train model using lasso
lasso_model <- glmnet::cv.glmnet(x_train, y_train, alpha = 1)

# Predict target values
train_predictions <- predict(lasso_model, x_train, s = lasso_model$lambda.min)
test_predictions  <- predict(lasso_model, x_test, s = lasso_model$lambda.min)

# Attach predictions to original data frames
train_final <- dplyr::bind_cols(train, prediction = train_predictions)
test_final  <- dplyr::bind_cols(test, prediction = test_predictions)

# Create submission data frame
submission <- test_final %>%
  dplyr::select(ID, target = prediction)

# Write to CSV file
write.csv(submission, 'santander_submission_lasso.csv', row.names = FALSE)
