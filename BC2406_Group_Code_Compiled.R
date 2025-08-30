# #Install Packages
# install.packages(dplyr)
# install.packages(tibble)
# install.packages(data.table)
# install.packages(ggplot2)
# install.packages(rpart)
# install.packages(glmnet)
# install.packages(caret)
# install.packages(xgboost)
# install.packages(rpart.plot)
# install.packages(pROC)
# install.packages(corrplot)
# install.packages(randomForest)

#Import Libraries
library(dplyr)
library(tibble)
library(data.table)
library(ggplot2)
library(rpart)
library(glmnet)
library(caret)
library(xgboost)
library(rpart.plot)
library(pROC)
library(corrplot)
library(randomForest)

# setwd("/Users/tzi/School/BC2406")

# Load the data
data <- read.csv("reckless_driving_dataset_V2.csv")

######################################################################################################
# Basic Models (Linear Regression, Logistic Regression, CART Model) Seed - 42
######################################################################################################


# 1. Data Preprocessing
# Convert categorical columns to factors, including `Adaptive_Cruise_Control_Activation`
categorical_cols <- c("Weather_Condition", "Road_Type", "Road_Condition", "Time_of_Day", "Vehicle_Type", 
                      "Adaptive_Cruise_Control_Activation")
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

# Convert Accident_Occurrence to a factor with levels "No" and "Yes"
data$Accident_Occurrence <- factor(data$Accident_Occurrence, levels = c(0, 1), labels = c("No", "Yes"))

# Prepare data including and excluding Near_Miss_Events
data_for_accident_with_miss <- data %>% select(-c(Intervention_Needed, Intervention_Level))
data_for_accident_no_miss <- data %>% select(-c(Intervention_Needed, Intervention_Level, Near_Miss_Events))

# Prepare data for Near_Miss_Events analysis
data_for_near_miss <- data %>% select(-c(Accident_Occurrence, Intervention_Needed, Intervention_Level))

# Train-test split (70-30)
set.seed(42)
train_index_accident_with_miss <- createDataPartition(data_for_accident_with_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_with_miss <- data_for_accident_with_miss[train_index_accident_with_miss, ]
test_data_accident_with_miss <- data_for_accident_with_miss[-train_index_accident_with_miss, ]

train_index_accident_no_miss <- createDataPartition(data_for_accident_no_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_no_miss <- data_for_accident_no_miss[train_index_accident_no_miss, ]
test_data_accident_no_miss <- data_for_accident_no_miss[-train_index_accident_no_miss, ]

train_index_near_miss <- createDataPartition(data_for_near_miss$Near_Miss_Events, p = 0.7, list = FALSE)
train_data_near_miss <- data_for_near_miss[train_index_near_miss, ]
test_data_near_miss <- data_for_near_miss[-train_index_near_miss, ]

# 2. Linear Regression for Near_Miss_Events
linear_model <- lm(Near_Miss_Events ~ ., data = train_data_near_miss)
summary(linear_model)

# Predicting Near Miss Events on test data
test_predictions <- predict(linear_model, test_data_near_miss)

# Calculating Mean Absolute Error (MAE)
mae <- mean(abs(test_predictions - test_data_near_miss$Near_Miss_Events))
print(paste("Mean Absolute Error:", mae))

# Calculating Mean Squared Error (MSE)
mse <- mean((test_predictions - test_data_near_miss$Near_Miss_Events)^2)
print(paste("Mean Squared Error:", mse))

# Calculating Root Mean Squared Error (RMSE)
rmse <- sqrt(mse)
print(paste("Root Mean Squared Error:", rmse))

# 3. Logistic Regression for Accident_Occurrence
# a) Including Near_Miss_Events
logistic_model_with_miss <- glm(Accident_Occurrence ~ ., data = train_data_accident_with_miss, family = binomial)
summary(logistic_model_with_miss)

# Evaluate Logistic Model with Near_Miss_Events
predictions_logistic_with_miss <- predict(logistic_model_with_miss, test_data_accident_with_miss, type = "response")
predicted_class_logistic_with_miss <- ifelse(predictions_logistic_with_miss > 0.5, "Yes", "No")
predicted_class_logistic_with_miss <- factor(predicted_class_logistic_with_miss, levels = c("No", "Yes"))
conf_matrix_logistic_with_miss <- confusionMatrix(predicted_class_logistic_with_miss, test_data_accident_with_miss$Accident_Occurrence)
print(conf_matrix_logistic_with_miss)

# b) Excluding Near_Miss_Events
logistic_model_no_miss <- glm(Accident_Occurrence ~ ., data = train_data_accident_no_miss, family = binomial)
summary(logistic_model_no_miss)

# Evaluate Logistic Model without Near_Miss_Events
predictions_logistic_no_miss <- predict(logistic_model_no_miss, test_data_accident_no_miss, type = "response")
predicted_class_logistic_no_miss <- ifelse(predictions_logistic_no_miss > 0.5, "Yes", "No")
predicted_class_logistic_no_miss <- factor(predicted_class_logistic_no_miss, levels = c("No", "Yes"))
conf_matrix_logistic_no_miss <- confusionMatrix(predicted_class_logistic_no_miss, test_data_accident_no_miss$Accident_Occurrence)
print(conf_matrix_logistic_no_miss)

# 4. CART Model for Accident_Occurrence
# a) Including Near_Miss_Events (using 1-SE rule for pruning)
cart_model_with_miss <- rpart(Accident_Occurrence ~ ., data = train_data_accident_with_miss, method = "class", control = rpart.control(cp = 0, minsplit = 2, xval = 10))
cv_results_with_miss <- cart_model_with_miss$cptable
one_se_cp_with_miss <- max(cv_results_with_miss[cv_results_with_miss[, "xerror"] <= (min(cv_results_with_miss[, "xerror"]) + cv_results_with_miss[which.min(cv_results_with_miss[, "xerror"]), "xstd"]), "CP"])
pruned_cart_model_with_miss <- prune(cart_model_with_miss, cp = one_se_cp_with_miss)
rpart.plot(pruned_cart_model_with_miss, main = "Pruned CART Model for Accident Occurrence (with Near_Miss_Events)", extra = 104)

# b) Excluding Near_Miss_Events (fixed cp, no pruning)
cart_model_no_miss <- rpart(Accident_Occurrence ~ ., data = train_data_accident_no_miss, method = "class", control = rpart.control(cp = 0.001, minsplit = 10, maxdepth = 5))
rpart.plot(cart_model_no_miss, main = "CART Model for Accident Occurrence (without Near_Miss_Events)", extra = 104)

# Evaluate CART Models
predictions_cart_with_miss <- predict(pruned_cart_model_with_miss, test_data_accident_with_miss, type = "class")
predictions_cart_with_miss <- factor(predictions_cart_with_miss, levels = c("No", "Yes"))
conf_matrix_cart_with_miss <- confusionMatrix(predictions_cart_with_miss, test_data_accident_with_miss$Accident_Occurrence)
print(conf_matrix_cart_with_miss)

predictions_cart_no_miss <- predict(cart_model_no_miss, test_data_accident_no_miss, type = "class")
predictions_cart_no_miss <- factor(predictions_cart_no_miss, levels = c("No", "Yes"))
conf_matrix_cart_no_miss <- confusionMatrix(predictions_cart_no_miss, test_data_accident_no_miss$Accident_Occurrence)
print(conf_matrix_cart_no_miss)

# 5. Visualizations

# Extract top significant variables for Near Miss Events, excluding the Intercept
coefficients <- summary(linear_model)$coefficients
coefficients <- coefficients[-1, ] # Remove Intercept
significant_vars <- coefficients[order(abs(coefficients[, "t value"]), decreasing = TRUE),]
print("Top variables influencing Near Miss Events:")
print(head(significant_vars, 5))

# Plotting Linear Regression Coefficients for Near Miss Events
significant_coefficients_df <- as.data.frame(significant_vars)
significant_coefficients_df$Variable <- rownames(significant_coefficients_df)

ggplot(significant_coefficients_df, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Linear Regression Coefficients for Near Miss Events", x = "Variable", y = "Coefficient") +
  theme_minimal()

# Plotting Logistic Regression Coefficients for Accident Occurrence (with and without Near_Miss_Events)
logistic_coefficients_with_miss_df <- as.data.frame(summary(logistic_model_with_miss)$coefficients)
logistic_coefficients_with_miss_df <- logistic_coefficients_with_miss_df[-1, ]  # Remove Intercept
logistic_coefficients_with_miss_df$Variable <- rownames(logistic_coefficients_with_miss_df)

ggplot(logistic_coefficients_with_miss_df, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Logistic Regression Coefficients for Accident Occurrence (with Near_Miss_Events)", x = "Variable", y = "Coefficient") +
  theme_minimal()

logistic_coefficients_no_miss_df <- as.data.frame(summary(logistic_model_no_miss)$coefficients)
logistic_coefficients_no_miss_df <- logistic_coefficients_no_miss_df[-1, ]  # Remove Intercept
logistic_coefficients_no_miss_df$Variable <- rownames(logistic_coefficients_no_miss_df)

ggplot(logistic_coefficients_no_miss_df, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Logistic Regression Coefficients for Accident Occurrence (without Near_Miss_Events)", x = "Variable", y = "Coefficient") +
  theme_minimal()



# =========================================
# 1. Univariate Analysis
# =========================================

# Histogram for Near_Miss_Events
ggplot(data, aes(x = Near_Miss_Events)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Near Miss Events", x = "Near Miss Events", y = "Frequency")

# Bar plot for Accident Occurrence
ggplot(data, aes(x = Accident_Occurrence)) +
  geom_bar(fill = "salmon", color = "black") +
  labs(title = "Distribution of Accident Occurrence", x = "Accident Occurrence", y = "Count")

# Box plots for Driver_Fatigue_Level, Speed, Acceleration_Events, and Braking_Events
boxplot_vars <- c("Driver_Fatigue_Level", "Speed", "Acceleration_Events", "Braking_Events")

for (var in boxplot_vars) {
  p <- ggplot(data, aes(y = .data[[var]])) +  # Use .data[[var]] for tidy evaluation
    geom_boxplot(fill = "lightgreen") +
    labs(title = paste("Boxplot of", var), y = var) +
    theme_minimal()
  
  print(p)  # Print the plot within the loop
}

# =========================================
# 2. Bivariate Analysis
# =========================================

# Box plot of Near_Miss_Events by Accident_Occurrence
ggplot(data, aes(x = Accident_Occurrence, y = Near_Miss_Events)) +
  geom_boxplot(fill = "purple", color = "black") +
  labs(title = "Near Miss Events by Accident Occurrence", x = "Accident Occurrence", y = "Near Miss Events")

# Scatter plot and correlation of Driver_Fatigue_Level vs Near_Miss_Events
ggplot(data, aes(x = Driver_Fatigue_Level, y = Near_Miss_Events)) +
  geom_point(alpha = 0.5) +
  labs(title = "Driver Fatigue Level vs Near Miss Events", x = "Driver Fatigue Level", y = "Near Miss Events")

# Box plots of Speed, Acceleration_Events, and Braking_Events by Accident_Occurrence
for (var in c("Speed", "Acceleration_Events", "Braking_Events")) {
  p <- ggplot(data, aes(x = Accident_Occurrence, y = .data[[var]])) +
    geom_boxplot(fill = "lightblue", color = "black") +
    labs(title = paste(var, "by Accident Occurrence"), x = "Accident Occurrence", y = var) +
    theme_minimal()
  
  print(p)
}

# Stacked bar plot of Accident Occurrence by Weather Condition and Road Type
ggplot(data, aes(x = Weather_Condition, fill = Accident_Occurrence)) +
  geom_bar(position = "fill") +
  labs(title = "Accident Occurrence by Weather Condition", x = "Weather Condition", y = "Proportion")

ggplot(data, aes(x = Road_Type, fill = Accident_Occurrence)) +
  geom_bar(position = "fill") +
  labs(title = "Accident Occurrence by Road Type", x = "Road Type", y = "Proportion")

# =========================================
# 3. Correlation Matrix for Continuous Variables
# =========================================

# Select continuous variables
continuous_vars <- c("Near_Miss_Events", "Driver_Fatigue_Level", "Speed", "Acceleration_Events", "Braking_Events",
                     "Visibility", "Traffic_Density", "Speed_Limit", "Driver_Age", "Driving_Experience", "Vehicle_Age")

# Calculate correlation matrix
cor_matrix <- cor(data[continuous_vars], use = "complete.obs")

# Plot correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix of Continuous Variables")

# =========================================
# Statistical Tests (optional, but useful for confirming associations)
# =========================================

# T-test for Near_Miss_Events by Accident_Occurrence
t_test_result <- t.test(Near_Miss_Events ~ Accident_Occurrence, data = data)
print(t_test_result)

# Chi-square test for Weather_Condition and Accident_Occurrence
chisq_test_weather <- chisq.test(table(data$Weather_Condition, data$Accident_Occurrence))
print(chisq_test_weather)

# Chi-square test for Road_Type and Accident_Occurrence
chisq_test_road_type <- chisq.test(table(data$Road_Type, data$Accident_Occurrence))
print(chisq_test_road_type)



# Function to plot coefficients with error bars
plot_coefficients <- function(model, title) {
  coefficients <- summary(model)$coefficients
  coefficients_df <- as.data.frame(coefficients)
  coefficients_df$Variable <- rownames(coefficients_df)
  coefficients_df$OddsRatio <- exp(coefficients_df$Estimate)
  coefficients_df$Lower_CI <- exp(coefficients_df$Estimate - 1.96 * coefficients_df$`Std. Error`)
  coefficients_df$Upper_CI <- exp(coefficients_df$Estimate + 1.96 * coefficients_df$`Std. Error`)
  
  ggplot(coefficients_df[-1, ], aes(x = reorder(Variable, OddsRatio), y = OddsRatio)) +
    geom_point() +
    geom_errorbar(aes(ymin = Lower_CI, ymax = Upper_CI), width = 0.2) +
    coord_flip() +
    labs(title = title, x = "Variable", y = "Odds Ratio (Exp(Coefficient))") +
    theme_minimal()
}

# Plot coefficients with error bars for both models
plot_with_miss <- plot_coefficients(logistic_model_with_miss, "Coefficients for Logistic Model with Near Miss Events")
plot_no_miss <- plot_coefficients(logistic_model_no_miss, "Coefficients for Logistic Model without Near Miss Events")

print(plot_with_miss)
print(plot_no_miss)

######################################################################################################
# Advanced Models (Random Forest, XGBoost) Seed - 42
######################################################################################################

#Reset data
data <- fread("reckless_driving_dataset_V2.csv")
data <- data %>%
  mutate(Speed_Exceed = ifelse(Speed > Speed_Limit, Speed - Speed_Limit, 0))
data <- data %>%
  mutate(Intervention_Needed=NULL,Intervention_Level=NULL)


# Converting character columns to factors
data$Weather_Condition <- factor(data$Weather_Condition)
data$Road_Type <- factor(data$Road_Type)
data$Road_Condition <- factor(data$Road_Condition)
data$Time_of_Day <- factor(data$Time_of_Day)
data$Vehicle_Type <- factor(data$Vehicle_Type)
data$Adaptive_Cruise_Control_Activation <- factor(data$Adaptive_Cruise_Control_Activation)
data$Accident_Occurrence <- factor(data$Accident_Occurrence, levels = c(0, 1), labels = c("No", "Yes"))

# Prepare data including and excluding Near_Miss_Events
data_for_accident_with_miss <- data 
data_for_accident_no_miss <- data %>% select(-c(
  Near_Miss_Events))

# Additional removal of variable (noise) after initial random forest variable importance
data_for_accident_no_miss <- data_for_accident_no_miss %>% select(-c(Vehicle_Age, Driving_Experience,
                                                                     Lane_Departure_Events, Speed_Exceed
))

# Checking to confirm the conversion
str(data)

# Train-test split (70-30)
set.seed(42)
train_index_accident_with_miss <- createDataPartition(data_for_accident_with_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_with_miss <- data_for_accident_with_miss[train_index_accident_with_miss, ]
test_data_accident_with_miss <- data_for_accident_with_miss[-train_index_accident_with_miss, ]

train_index_accident_no_miss <- createDataPartition(data_for_accident_no_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_no_miss <- data_for_accident_no_miss[train_index_accident_no_miss, ]
test_data_accident_no_miss <- data_for_accident_no_miss[-train_index_accident_no_miss, ]
#data_for_near_miss$Near_Miss_Events <- factor(ifelse(data_for_near_miss$Near_Miss_Events > 0, "Yes", "No"))

# Then split the data
#train_index_near_miss <- createDataPartition(data_for_near_miss$Near_Miss_Events, p = 0.7, list = FALSE)
#train_data_near_miss <- data_for_near_miss[train_index_near_miss, ]
#test_data_near_miss <- data_for_near_miss[-train_index_near_miss, ]

# Check class distribution in the dataset
table(train_data_accident_no_miss$Accident_Occurrence)

# Alternatively, you can get proportions
prop.table(table(train_data_accident_no_miss$Accident_Occurrence))

######################################################################################################
# Random Forest
######################################################################################################
# Train an RF model (Accident ~ With near miss)
rf_model <- randomForest(Accident_Occurrence ~ ., data = train_data_accident_with_miss, ntree = 200, mtry = 6, importance = TRUE)

# Predictions
predictions <- predict(rf_model, newdata = test_data_accident_with_miss)
# Confusion matrix and accuracy
confusion_matrix <- table(test_data_accident_with_miss$Accident_Occurrence, predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

print(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Ensure both actual and predicted values are factors with the same levels
test_data_accident_with_miss$Accident_Occurrence <- factor(test_data_accident_with_miss$Accident_Occurrence, levels = c("No", "Yes"))
predictions <- factor(predictions, levels = c("No", "Yes"))

# Generate the confusion matrix with statistics
confusion_matrix_results_2 <- confusionMatrix(predictions, test_data_accident_with_miss$Accident_Occurrence)

# Print the results
print(confusion_matrix_results_2)

importance(rf_model)
varImpPlot(rf_model)

######################################################################################################
# Train an RF model (Accident ~ With no miss)
rf_model_nomiss <- randomForest(Accident_Occurrence ~ ., data = train_data_accident_no_miss, ntree = 250, mtry = 11, importance = TRUE)

predictions_nomiss <- predict(rf_model_nomiss, newdata = test_data_accident_no_miss)

confusion_matrix_no <- table(test_data_accident_no_miss$Accident_Occurrence, predictions_nomiss)
accuracy_no <- sum(diag(confusion_matrix_no)) / sum(confusion_matrix_no)

print(confusion_matrix_no)
print(paste("Accuracy:", round(accuracy_no * 100, 2), "%"))

# Ensure both actual and predicted values are factors with the same levels
test_data_accident_no_miss$Accident_Occurrence <- factor(test_data_accident_no_miss$Accident_Occurrence, levels = c("No", "Yes"))
predictions_nomiss <- factor(predictions_nomiss, levels = c("No", "Yes"))

# Generate the confusion matrix with statistics
confusion_matrix_results <- confusionMatrix(predictions_nomiss, test_data_accident_no_miss$Accident_Occurrence)

# Print the results
print(confusion_matrix_results)

#rf_model_nearmiss <- randomForest(Near_Miss_Events ~ ., data = train_data_near_miss, ntree = 200, mtry = 6, importance = TRUE)

#predictions_nearmiss <- predict(rf_model_nearmiss, newdata = test_data_near_miss)
#confusion_matrix_near <- table(test_data_near_miss$Near_Miss_Events, predictions_nearmiss)
#accuracy_near <- sum(diag(confusion_matrix_near)) / sum(confusion_matrix_near)

#print(confusion_matrix_near)
#print(paste("Accuracy:", round(accuracy_near * 100, 2), "%"))


# Variable importance

importance(rf_model_nomiss)
varImpPlot(rf_model_nomiss)
#########################################################################################################
# XGBoost
#########################################################################################################
data_for_accident_no_miss <- data %>% select(-c(
  Near_Miss_Events))

train_index_accident_with_miss <- createDataPartition(data_for_accident_with_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_with_miss <- data_for_accident_with_miss[train_index_accident_with_miss, ]
test_data_accident_with_miss <- data_for_accident_with_miss[-train_index_accident_with_miss, ]

train_index_accident_no_miss <- createDataPartition(data_for_accident_no_miss$Accident_Occurrence, p = 0.7, list = FALSE)
train_data_accident_no_miss <- data_for_accident_no_miss[train_index_accident_no_miss, ]
test_data_accident_no_miss <- data_for_accident_no_miss[-train_index_accident_no_miss, ]


train_data_accident_with_miss$Accident_Occurrence <- as.numeric(train_data_accident_with_miss$Accident_Occurrence) - 1
test_data_accident_with_miss$Accident_Occurrence <- as.numeric(test_data_accident_with_miss$Accident_Occurrence) - 1

train_data_accident_no_miss$Accident_Occurrence <- as.numeric(train_data_accident_no_miss$Accident_Occurrence) - 1
test_data_accident_no_miss$Accident_Occurrence <- as.numeric(test_data_accident_no_miss$Accident_Occurrence) - 1

# Prepare training and testing matrices
x_train <- model.matrix(Accident_Occurrence ~ Speed + Driver_Age + Traffic_Density + Visibility + 
                          Driver_Fatigue_Level + Speed_Limit + Acceleration_Events + 
                          Braking_Events + Steering_Input + Distraction_Events + 
                          Near_Miss_Events + Lane_Departure_Events + Adaptive_Cruise_Control_Activation + 
                          Driving_Experience + Vehicle_Age, data = train_data_accident_with_miss)[, -1]
y_train <- train_data_accident_with_miss$Accident_Occurrence

x_test <- model.matrix(Accident_Occurrence ~ Speed + Driver_Age + Traffic_Density + Visibility + 
                         Driver_Fatigue_Level + Speed_Limit + Acceleration_Events + 
                         Braking_Events + Steering_Input + Distraction_Events + 
                         Near_Miss_Events + Lane_Departure_Events + Adaptive_Cruise_Control_Activation + 
                         Driving_Experience + Vehicle_Age, data = test_data_accident_with_miss)[, -1]
y_test <- test_data_accident_with_miss$Accident_Occurrence

# Prepare training and testing matrices
x_train_no <- model.matrix(Accident_Occurrence ~ Speed + Driver_Age + Traffic_Density + Visibility + 
                          Driver_Fatigue_Level + Speed_Limit + Acceleration_Events + 
                          Braking_Events + Steering_Input + Distraction_Events + 
                          Lane_Departure_Events + Adaptive_Cruise_Control_Activation + 
                          Driving_Experience + Vehicle_Age, data = train_data_accident_no_miss)[, -1]
y_train_no <- train_data_accident_no_miss$Accident_Occurrence

x_test_no <- model.matrix(Accident_Occurrence ~ Speed + Driver_Age + Traffic_Density + Visibility + 
                         Driver_Fatigue_Level + Speed_Limit + Acceleration_Events + 
                         Braking_Events + Steering_Input + Distraction_Events + 
                         Lane_Departure_Events + Adaptive_Cruise_Control_Activation + 
                         Driving_Experience + Vehicle_Age, data = test_data_accident_no_miss)[, -1]
y_test_no <- test_data_accident_no_miss$Accident_Occurrence

############################################################################################################


# Train an XGBoost model (Accident ~ With near miss)
xgb_model <- xgboost(data = x_train, 
                     label = y_train, 
                     nrounds = 200, 
                     max_depth = 6, 
                     eta = 0.1, 
                     objective = "binary:logistic",
                     colsample_bytree = 0.8, 
                     subsample = 0.8,
                     verbose = 0)  # suppress output

# Make predictions on the test set
pred_prob <- predict(xgb_model, x_test)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)  # Convert probabilities to 0 or 1

# Convert predicted and actual values to factors with "No" and "Yes" levels
pred_class <- factor(pred_class, levels = c(0, 1), labels = c("No", "Yes"))
y_test <- factor(y_test, levels = c(0, 1), labels = c("No", "Yes"))

# Generate the confusion matrix with updated levels
conf_matrix <- confusionMatrix(pred_class, y_test, positive = "Yes")

# Display the confusion matrix and other statistics
print(conf_matrix)

############################################################################################################
# Train an XGBoost model (Accident ~ No near miss)
xgb_model_no <- xgboost(data = x_train_no, 
                     label = y_train_no, 
                     nrounds = 200, 
                     max_depth = 6, 
                     eta = 0.1, 
                     objective = "binary:logistic",
                     colsample_bytree = 0.8, 
                     subsample = 0.8,
                     verbose = 0)  # suppress output

# Make predictions on the test set
pred_prob_no <- predict(xgb_model_no, x_test_no)
pred_class_no <- ifelse(pred_prob_no > 0.5, 1, 0)  # Convert probabilities to 0 or 1

# Convert predicted and actual values to factors with "No" and "Yes" levels
pred_class_no <- factor(pred_class_no, levels = c(0, 1), labels = c("No", "Yes"))
y_test_no <- factor(y_test_no, levels = c(0, 1), labels = c("No", "Yes"))

# Generate the confusion matrix with updated levels
conf_matrix_no <- confusionMatrix(pred_class_no, y_test_no, positive = "Yes")

# Display the confusion matrix and other statistics
print(conf_matrix_no)

