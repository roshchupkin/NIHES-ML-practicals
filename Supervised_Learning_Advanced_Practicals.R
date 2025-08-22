# -----------------------------------------------------------------------------
# Supervised Learning Advanced Practicals:
# 1. Cross-Validation
# 2. Hyperparameter Tuning  
# 3. Feature Engineering & Selection
#
# Objective: To build upon the foundational modeling skills from the first
# practical by learning more robust evaluation and optimization techniques.
#
# Instructions:
# 1. Make sure you have the 'data_cleaned.csv' file in your R working directory.
# 2. Read the comments carefully to understand the purpose of each step.
# 3. Fill in the code in the sections marked with '# TODO:'. Hints are provided.
# 4. Run the script section by section to see the outputs.
# -----------------------------------------------------------------------------


# --- Section 0: Setup - Load Libraries and Data ---

# We'll start by loading all the necessary libraries for the entire script
# and loading the cleaned dataset you prepared in the first practical.

# ============================================================================
# LOADING REQUIRED LIBRARIES
# ============================================================================
# tidyverse: Collection of R packages for data science (includes ggplot2, dplyr, etc.)
# This provides tools for data manipulation, visualization, and analysis
library(tidyverse)

# caret: Classification And REgression Training - main package for ML workflows
# Provides unified interface for training and evaluating models with cross-validation
library(caret)

# rpart: Recursive Partitioning and Regression Trees
# This is the engine that caret uses when method = "rpart" (decision trees)
library(rpart)

# randomForest: Implementation of Random Forest algorithm
# This is the engine that caret uses when method = "rf" (random forests)
library(randomForest)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
# Load the cleaned data from Practical 1
# This file should be in the same folder as this R script.
# read_csv() is from the readr package (part of tidyverse) and is faster than base R's read.csv()
df <- read_csv("data_cleaned.csv")

# ============================================================================
# DATA TYPE CONVERSION FOR CLASSIFICATION
# ============================================================================
# For classification models, it's best practice to convert the outcome
# variable to a factor. This helps `caret` automatically handle it correctly.
# Factors tell R that this is a categorical variable, not numeric
# This is CRUCIAL for classification tasks in R
df$diabetes <- as.factor(df$diabetes)

# ============================================================================
# QUICK DATA EXPLORATION (Optional but recommended)
# ============================================================================
# Let's take a quick look at our data to make sure everything loaded correctly
print("Data dimensions:")
print(dim(df))  # Shows number of rows and columns

print("First few rows:")
print(head(df))  # Shows first 6 rows

print("Class balance:")
print(table(df$diabetes))  # Shows count of each class
print(prop.table(table(df$diabetes)))  # Shows proportions



# --- Section 1: Cross-Validation ---

# Objective: Understand the limitations of a single train/test split and
# learn how to use k-fold cross-validation for more robust model evaluation.

### Why Cross-Validation?
# In the previous practical, we split our data *once*. This is fast but can be
# misleading. What if our single test set was unusually easy or hard to predict?
# Our performance estimate would be overly optimistic or pessimistic.
#
# k-Fold Cross-Validation (CV) solves this by splitting the data into 'k' folds
# (e.g., k=10), training on 9 folds and testing on 1, and repeating this process
# 10 times. The final performance is the average across all 10 folds, giving
# a much more reliable estimate of how the model will perform on new, unseen data.

# ============================================================================
# THEORETICAL BACKGROUND: CROSS-VALIDATION
# ============================================================================
# Cross-validation addresses several key problems with single train/test splits:
#
# 1. VARIANCE REDUCTION: Single splits can give very different results depending
#    on which data points end up in training vs test sets
# 2. BIAS REDUCTION: Using all data for both training and testing (in different folds)
#    gives more reliable estimates
# 3. OVERFITTING DETECTION: CV helps identify if a model is overfitting to the training data
# 4. FAIR COMPARISON: All models are evaluated on the same folds, ensuring fair comparison
#
# HOW K-FOLD CV WORKS:
# 1. Data is randomly split into k equal parts (folds)
# 2. Model is trained on k-1 folds, tested on 1 fold
# 3. This is repeated k times, each fold serving as test set once
# 4. Results are averaged across all k iterations
# 5. Standard deviation is also calculated to assess uncertainty

### Implementing 10-Fold Cross-Validation with `caret`

# The `caret` package makes CV easy. We first define our "rules" for training
# using the trainControl() function.

# ============================================================================
# CONFIGURING CROSS-VALIDATION PARAMETERS
# ============================================================================
# trainControl() creates an object that tells caret HOW to perform resampling
train_control <- trainControl(
  method = "cv",    # Method: "cv" = k-fold cross-validation
  number = 10       # Number of folds (k = 10)
)

# ============================================================================
# UNDERSTANDING THE PARAMETERS:
# ============================================================================
# method = "cv": Standard k-fold cross-validation
#   - Alternative methods: "repeatedcv" (repeated k-fold), "boot" (bootstrap)
#   - "LOOCV" (Leave-One-Out CV) for very small datasets
#
# number = 10: Number of folds
#   - k = 10: Good balance between computational cost and reliability
#   - k = 5: Faster but less reliable
#   - k = 20: More reliable but slower


# -- Task 1.1: Train a Logistic Regression model with 10-fold CV --

# We'll use the `train()` function from caret, which is a powerful, unified
# interface for training many different types of models.

# ============================================================================
# LOGISTIC REGRESSION THEORY
# ============================================================================
# Logistic regression is a linear model for binary classification
# - Uses sigmoid function: P(Y=1) = 1 / (1 + e^(-z)) where z = β₀ + β₁X₁ + ... + βₚXₚ
# - Assumes linear relationship between predictors and log-odds
# - Good baseline model, interpretable coefficients
# - Assumes independence of observations

set.seed(42) # for reproducibility - ensures same random splits every time

# TODO: Train the logistic regression model using the train() function.
# HINT 1: The formula for predicting diabetes based on all other variables is `diabetes ~ .`
# HINT 2: The method for logistic regression in caret is "glm".
# HINT 3: You also need to specify `family = "binomial"` for logistic regression.
# HINT 4: Use the `trControl` argument to pass our `train_control` object.

logistic_model_cv <- # YOUR CODE HERE
  
  # ============================================================================
  # MODEL OUTPUT INTERPRETATION
  # ============================================================================
  # The output shows:
  # - Cross-validation results across all 10 folds
  # - Mean and standard deviation of performance metrics (Accuracy, Kappa)
  # - Best model parameters (if any tuning was done)
  # - Final model object that can be used for predictions
  
  # Print the results to see the average accuracy across the 10 folds
  print(logistic_model_cv)


# -- Task 1.2: Train a Decision Tree model with 10-fold CV --

# The process is very similar, we just change the `method` argument.

# ============================================================================
# DECISION TREE THEORY
# ============================================================================
# Decision trees recursively split data based on predictor values
# - Non-parametric method (no assumptions about data distribution)
# - Creates tree structure with decision rules
# - Easy to interpret and visualize
# - Can capture non-linear relationships
# - Prone to overfitting (hence cross-validation is crucial)

set.seed(42)

# TODO: Train the decision tree model using the train() function.
# HINT: The method for a decision tree in caret is "rpart".

tree_model_cv <- # YOUR CODE HERE
  
  # Print the results
  print(tree_model_cv)


# -- Task 1.3: Train a Random Forest model with 10-fold CV --

# The method for Random Forest is "rf".
# NOTE: This will take a bit longer to run!

# ============================================================================
# RANDOM FOREST THEORY
# ============================================================================
# Random Forest builds multiple decision trees and averages their predictions
# - Ensemble method (combines multiple models)
# - Each tree is trained on a bootstrap sample with random feature selection
# - Reduces overfitting through averaging
# - Handles non-linear relationships well
# - Provides feature importance measures
# - Generally good performance across many datasets
# - Less interpretable than single trees

set.seed(42)

# TODO: Train the random forest model using the train() function.
# HINT: The method is "rf".

forest_model_cv <- # YOUR CODE HERE
  
  # Print the results
  print(forest_model_cv)


### Comparing Models
# `caret` makes it easy to compare the performance of models trained
# with the same cross-validation scheme.

# ============================================================================
# FAIR MODEL COMPARISON USING IDENTICAL CV FOLDS
# ============================================================================
# This is a CRITICAL step for fair comparison!
# All models are evaluated on exactly the same cross-validation folds
# This eliminates fold-to-fold variation as a source of differences

# Create a list of the trained models
results <- resamples(list(Logistic = logistic_model_cv,
                          Tree = tree_model_cv,
                          Forest = forest_model_cv))

# ============================================================================
# COMPARISON SUMMARY STATISTICS
# ============================================================================
# Shows mean, median, min, max, and quartiles for each model
# Look for:
# - Which model has highest mean performance
# - Which model has lowest variance (most consistent)
# - Whether differences are statistically significant
summary(results)

# ============================================================================
# VISUALIZING MODEL COMPARISONS
# ============================================================================
# Box-and-whisker plots show the distribution of performance across CV folds
# - Box: Interquartile range (25th to 75th percentile)
# - Whiskers: Extend to most extreme non-outlier points
# - Points: Outliers
# - Higher boxes = better performance
# - Smaller boxes = more consistent performance
bwplot(results) # Box-and-whisker plots

# Dot plots show mean performance with confidence intervals
# - Dots: Mean performance across CV folds
# - Lines: Confidence intervals
# - Models with non-overlapping CIs are significantly different
dotplot(results) # Dot plots


# --- Section 2: Hyperparameter Tuning ---

# Objective: Understand what hyperparameters are and learn how to perform
# grid search to find the optimal settings for a model.

### What are Hyperparameters?
# These are the "knobs" or "settings" of a model that we set *before* training.
# For a Decision Tree, the 'complexity parameter' (`cp`) controls how complex the tree can get.
# For a Random Forest, 'mtry' controls how many features are considered at each split.
# Finding the best values for these hyperparameters can significantly improve model performance.

# ============================================================================
# HYPERPARAMETER TUNING THEORY
# ============================================================================
# Hyperparameters are model settings that control the learning process
# They are NOT learned from data but set by the user before training
#
# WHY TUNE HYPERPARAMETERS?
# - Can significantly improve model performance
# - Helps prevent overfitting or underfitting
# - Different datasets may need different settings
#
# COMMON TUNING METHODS:
# 1. Grid Search: Test all combinations of parameter values
# 2. Random Search: Randomly sample parameter combinations
# 3. Bayesian Optimization: Use previous results to guide search
#
# TRADE-OFFS:
# - More parameter combinations = better results but slower computation
# - Need to balance exploration vs computational cost


# -- Task 2.1: Tune the Complexity Parameter (`cp`) of a Decision Tree --

# We will create a "tuning grid" of different `cp` values to test. `caret` will
# automatically use our 10-fold CV to evaluate each one and pick the best.

# ============================================================================
# COMPLEXITY PARAMETER (cp) EXPLANATION
# ============================================================================
# cp controls tree complexity and helps prevent overfitting:
# - cp = 0: Full tree (likely overfitting)
# - cp > 0: Pruned tree (reduced complexity)
# - Higher cp = simpler tree = less overfitting but potentially underfitting
# - Typical range: 0.001 to 0.1
# - Rule of thumb: Start with 0.01 and adjust based on results

set.seed(42)

# TODO: Create the tuning grid for the 'cp' hyperparameter.
# We want to test a sequence of values from 0.01 to 0.1 in steps of 0.01.
# HINT: Use the `expand.grid()` function. The grid should have one column named `cp`.
tree_grid <- # YOUR CODE HERE
  
  # TODO: Train the decision tree model, this time including the `tuneGrid` argument.
  tuned_tree_model <- train(diabetes ~ .,
                            data = df,
                            method = "rpart",
                            trControl = train_control,
                            tuneGrid = # YOUR CODE HERE
  )

# ============================================================================
# TUNING RESULTS INTERPRETATION
# ============================================================================
# The output shows:
# - Best cp value found
# - Performance metrics for each cp value tested
# - Final model with optimal parameters
print(tuned_tree_model)

# ============================================================================
# VISUALIZING TUNING RESULTS
# ============================================================================
# Plot shows how model performance changes with different cp values
# Look for the "elbow" where increasing complexity doesn't improve performance
# This helps understand the bias-variance trade-off
plot(tuned_tree_model)


# -- Task 2.2: Tune the `mtry` of a Random Forest --

# Now, let's tune the `mtry` hyperparameter for our Random Forest model.
# A good rule of thumb is to test values around the square root of the number of predictors.
# We have 16 predictors, so sqrt(16) = 4. Let's test values from 2 to 8.

# ============================================================================
# MTRY PARAMETER EXPLANATION
# ============================================================================
# mtry: Number of variables randomly sampled at each split
# - mtry = 1: Use only 1 variable per split (like bagging)
# - mtry = sqrt(p): Common default (p = number of predictors)
# - mtry = p: Use all variables (like single tree)
# - Lower mtry = more randomness = less overfitting but potentially underfitting
# - Higher mtry = less randomness = more overfitting risk

set.seed(42)

# TODO: Create the tuning grid for the 'mtry' hyperparameter.
# HINT: Use `expand.grid()` and test the integer values 2, 3, 4, 5, 6, 7, 8.
forest_grid <- # YOUR CODE HERE
  
  # TODO: Train the random forest model with the tuning grid.
  # NOTE: This will take longer to run than the decision tree!
  tuned_forest_model <- train(diabetes ~ .,
                              data = df,
                              method = "rf",
                              trControl = train_control,
                              tuneGrid = # YOUR CODE HERE
  )

# Print the results
print(tuned_forest_model)

# Plot the results
plot(tuned_forest_model)



# --- Section 3: Feature Engineering & Selection ---

# Objective: Learn how creating new features and selecting the most important
# ones can impact model performance.

# ============================================================================
# FEATURE ENGINEERING THEORY
# ============================================================================
# Feature engineering is the process of creating new features from existing data
# WHY FEATURE ENGINEERING?
# - Can capture non-linear relationships
# - Can encode domain knowledge
# - Can improve model interpretability
# - Can help with data limitations
#
# COMMON TECHNIQUES:
# 1. Binning: Convert continuous variables to categorical
# 2. Polynomial features: Create interaction terms
# 3. Domain-specific features: Use expert knowledge
# 4. Time-based features: Extract date/time components

### Part 1: Feature Engineering - Creating an 'age_group' Feature

# Sometimes, creating new features from existing ones can help the model.
# Let's convert the continuous 'age' variable into a categorical 'age_group'.

# ============================================================================
# AGE GROUPING RATIONALE
# ============================================================================
# Converting age to groups can help because:
# - Age effects on diabetes may be non-linear
# - Groups can capture threshold effects
# - Reduces noise in the age variable
# - Makes the model more interpretable
# - Can help with small sample sizes in extreme age groups

df_engineered <- df %>%
  mutate(
    # ============================================================================
    # CUT() FUNCTION EXPLANATION
    # ============================================================================
    # cut() converts continuous variables to categorical
    # breaks: defines the boundaries for each group
    # labels: provides meaningful names for each group
    # right = FALSE: intervals are [a,b) instead of (a,b]
    age_group = cut(age,
                    breaks = c(0, 30, 50, 70, 100),  # Age boundaries
                    labels = c("Young", "MiddleAged", "Senior", "Elderly"),  # Group names
                    right = FALSE)  # Use [a,b) intervals
  )

# ============================================================================
# ONE-HOT ENCODING EXPLANATION
# ============================================================================
# One-hot encoding converts categorical variables to binary columns
# WHY ONE-HOT ENCODE?
# - Most ML algorithms can't handle categorical variables directly
# - Creates separate binary columns for each category
# - Example: age_group becomes 4 columns (Young_Yes/No, MiddleAged_Yes/No, etc.)
# - Avoids ordinal assumptions (Young < MiddleAged < Senior < Elderly)

# Now we need to one-hot encode this new categorical feature so the model can use it.
# The `caret` package has a function called `dummyVars` which is great for this.

# TODO: Create the dummy variable formula and apply it.
# HINT 1: The formula should be `~ age_group`
# HINT 2: Use the `dummyVars()` function with the formula and data.
# HINT 3: Use the `predict()` function on the result of `dummyVars()` to create the new columns.

dummies <- # YOUR CODE HERE
  age_group_encoded <- # YOUR CODE HERE
  
  # ============================================================================
  # COMBINING ENGINEERED FEATURES
  # ============================================================================
  # cbind() combines dataframes by columns
  # select(-age_group) removes the original categorical column
  # This prevents redundancy (having both original and encoded versions)
  
  # Combine the new encoded columns with our dataframe
  df_engineered <- cbind(df_engineered, age_group_encoded) %>%
  select(-age_group) # Remove the original age_group column

# -- Task 3.1: Re-train and Evaluate a Model with the New Feature --
# Let's train a Random Forest to see if this new feature helps. We should
# remove the original 'age' column to avoid giving the model redundant information.

# ============================================================================
# AVOIDING REDUNDANCY
# ============================================================================
# We remove the original 'age' column because:
# - The new age_group features already contain age information
# - Having both could lead to multicollinearity
# - The model might give too much weight to age-related features
# - Simpler models are often more interpretable

set.seed(42)

# TODO: Train a Random Forest model on the `df_engineered` dataframe.
# HINT: The formula should be `diabetes ~ . - age` to predict diabetes using all
# variables *except* the original age column.

model_with_age_group <- train(# YOUR CODE HERE
)

# Print the results and compare to the previous Random Forest model.
print(model_with_age_group)


### Part 2: Feature Selection with Recursive Feature Elimination (RFE)

# RFE is a technique that helps us find the most important features by
# iteratively building models and discarding the weakest features.

# ============================================================================
# RECURSIVE FEATURE ELIMINATION (RFE) THEORY
# ============================================================================
# RFE is a wrapper method for feature selection that:
# 1. Trains a model with all features
# 2. Ranks features by importance
# 3. Removes the least important feature(s)
# 4. Repeats until desired number of features remains
#
# ADVANTAGES:
# - Finds optimal feature subset
# - Reduces overfitting
# - Improves interpretability
# - Can improve performance
#
# DISADVANTAGES:
# - Computationally expensive
# - May not find global optimum
# - Depends on the base model used

set.seed(42)

# ============================================================================
# RFE CONFIGURATION
# ============================================================================
# rfeControl() defines how RFE should work:
# - functions = rfFuncs: Use Random Forest for feature ranking
# - method = "cv": Use cross-validation for evaluation
# - number = 10: Use 10-fold cross-validation
rfe_control <- rfeControl(functions = rfFuncs,
                          method = "cv",
                          number = 10)

# ============================================================================
# DATA PREPARATION FOR RFE
# ============================================================================
# RFE needs separate predictor (X) and outcome (y) variables
# We exclude non-predictive columns like subject_id and checkup_date
# These columns don't help predict diabetes and could confuse the algorithm

# Separate predictors (X) and outcome (y)
# We exclude the target (diabetes) and non-predictive columns like subject_id and checkup_date
predictors <- df %>% select(-diabetes, -subject_id, -checkup_date)
outcome <- df$diabetes


# -- Task 3.2: Run the RFE algorithm --
# NOTE: This can take several minutes to run!

# TODO: Use the `rfe()` function to find the best features.
# HINT: The arguments are `x = predictors`, `y = outcome`, `sizes = c(...)`, and `rfeControl = rfe_control`.
# For `sizes`, let's test using the top 5, 10, and 15 features.
rfe_results <- rfe(# YOUR CODE HERE
)

# ============================================================================
# RFE RESULTS INTERPRETATION
# ============================================================================
# The output shows:
# - Best number of features
# - Performance metrics for each feature set size
# - List of selected features
# - Ranking of feature importance

# Print the results and see which feature set size performed best
print(rfe_results)

# List the chosen predictors for the best size
predictors(rfe_results)

# ============================================================================
# VISUALIZING RFE RESULTS
# ============================================================================
# Plot shows performance vs number of features
# Look for the "elbow" where adding more features doesn't help
# This helps understand the trade-off between complexity and performance
plot(rfe_results, type = c("g", "o"))


# -- Task 3.3: Train a Final Model with Selected Features --
# Now that RFE has identified the top predictors, let's train a model
# using *only* these features to see how it performs.

# ============================================================================
# FEATURE SELECTION BENEFITS
# ============================================================================
# Using only the top features can:
# - Reduce overfitting
# - Improve interpretability
# - Speed up training and prediction
# - Focus on the most important variables
# - Sometimes improve performance by removing noise

# Get the names of the top 5 predictors identified by RFE
top_5_predictors <- head(predictors(rfe_results), 5)
print(paste("Top 5 predictors:", paste(top_5_predictors, collapse=", ")))


# Create a new dataframe with only these predictors and the outcome
df_rfe <- df[c(top_5_predictors, "diabetes")]


# TODO: Train a Random Forest model using this new, smaller `df_rfe` dataset.
set.seed(42)

model_with_rfe <- train(diabetes ~ .,
                        data = df_rfe,
                        method = "rf",
                        trControl = train_control)

# Print the results
print(model_with_rfe)


# --- End of Practicals: Discussion Questions ---

# ============================================================================
# REFLECTION AND DISCUSSION
# ============================================================================
# These questions help you think critically about what you've learned
# and how to apply these techniques in real-world scenarios

# 1. Did cross-validation give you a different "best" model than the simple train/test split? Why is the CV result more trustworthy?
#    - Think about variance in performance estimates
#    - Consider the reliability of single splits
#    - Reflect on the importance of robust evaluation

# 2. Did hyperparameter tuning provide a significant boost in performance for the Decision Tree or Random Forest?
#    - Compare tuned vs untuned performance
#    - Consider the computational cost vs performance gain
#    - Think about when tuning is most valuable

# 3. What were the top 5 predictors identified by RFE? Do these make sense from a biological/health perspective?
#    - Consider clinical relevance
#    - Think about known risk factors for diabetes
#    - Reflect on the value of interpretable models

# 4. How did the model with only 5 features perform compared to the model with all features? What are the advantages of using a simpler model, even if accuracy is slightly lower?
#    - Consider interpretability vs performance trade-offs
#    - Think about practical deployment considerations
#    - Reflect on the value of parsimony in modeling

# ============================================================================
# TROUBLESHOOTING TIPS
# ============================================================================
# Common issues and solutions:
#
# 1. "Error: could not find function 'train'"
#    - Make sure you've loaded the caret library
#    - Check that caret is installed: install.packages("caret")
#
# 2. "Error: object 'df' not found"
#    - Check that your data file is in the working directory
#    - Use getwd() to see current directory
#    - Use setwd() to change directory if needed
#
# 3. "Error: invalid formula"
#    - Make sure diabetes is a factor: df$diabetes <- as.factor(df$diabetes)
#    - Check for missing values in your data
#
# 4. RFE takes too long
#    - Reduce the number of features to test in 'sizes'
#    - Use fewer CV folds (number = 5 instead of 10)
#    - Consider using a faster base model
#
# 5. Poor model performance
#    - Check class balance in your outcome variable
#    - Consider using different evaluation metrics (ROC AUC for imbalanced data)
#    - Try different algorithms or feature engineering approaches