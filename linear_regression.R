# Load packages
library(tidyverse)

# Read in data
train <- readr::read_csv('kaggle_data/train.csv.zip')
test  <- readr::read_csv('kaggle_data/test.csv.zip')

# Identify most highly correlated variables
correlations <- train %>%
  dplyr::select_if(is.numeric) %>%
  cor() %>%
  as.data.frame() %>%
  dplyr::mutate(field = rownames(.)) %>%
  dplyr::select(field, target) %>%
  dplyr::arrange(dplyr::desc(abs(target))) %>%
  dplyr::filter(field != 'target')

# Use top n variables
variables_to_use <- correlations %>%
  dplyr::top_n(20, target) %>%
  dplyr::pull(field)

# Identify median values of training set
medians <- new_train %>%
  dplyr::select_if(is.numeric) %>%
  dplyr::summarize_all(funs(median), na.rm = TRUE)

# Specify features and response variable
new_train <- train %>% 
  dplyr::select(target, dplyr::one_of(variables_to_use)) %>%
  tidyr::replace_na(medians)

new_test <- test %>% 
  dplyr::select(dplyr::one_of(variables_to_use)) %>%
  tidyr::replace_na(medians)

# Train model using lasso
regression_model <- lm(target ~ ., data = new_train)

# Predict target values
train_predictions <- predict(regression_model, new_train)
test_predictions  <- predict(regression_model, new_test)

# Attach predictions to original data frames
train_final <- dplyr::bind_cols(train, prediction = train_predictions)
test_final  <- dplyr::bind_cols(test, prediction = test_predictions)

# Create submission data frame
submission <- test_final %>%
  dplyr::select(ID, target = prediction) %>%
  dplyr::mutate(target = ifelse(target < 0, 0, target))

# Write to CSV file
write.csv(submission, 'santander_submission_linear_regression.csv', row.names = FALSE)
