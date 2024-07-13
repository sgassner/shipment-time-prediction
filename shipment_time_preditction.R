```r
# Load datasets
load("olist.Rdata")
load("olist_predict.Rdata")

# Load packages
library(tidyverse)
library(corrplot)
library(glmnet)
library(fastDummies)
library(moments)
library(grf)
library(knitr)
library(reshape2)
library(gridExtra)

############################################################
# Data Cleaning
############################################################

# Look at the structure of both data sets
str(olist)
str(olist_predict)

# Look at the summary statistics of both data sets
summary(olist)
summary(olist_predict)

# Remove Missing Values
olist <- olist[complete.cases(olist), ]
olist_predict <- olist_predict[complete.cases(olist_predict), ]

# Remove Unrealistic Values
olist <- olist %>% filter(weight_g != 0, freight_rate != 0, distance !=0)
olist <- olist %>% filter(shippingtime > 0)
olist_predict <- olist_predict %>% filter(weight_g != 0, freight_rate != 0, distance !=0)

# Count observations for each product category
n_product_categories <- olist %>% group_by(product_category_major) %>% 
  summarize(n = n()) %>% arrange(n)

# Find categories with at least 20 observations
n_categories_above_20 <- n_product_categories %>% filter(n>=20)

# Delete observations with less than 20 observations in both data sets
olist <- olist[olist$product_category_major %in% n_categories_above_20$product_category_major, ]
olist_predict <- olist_predict[olist_predict$product_category_major %in% n_categories_above_20$product_category_major, ]

# Correlation Plot
cor <- round(cor(olist[,c(2:15)]),2)
corrplot(cor)

# Remove variables with high correlation
olist <- olist %>% select(-deliverymonth)
olist_predict <- olist_predict %>% select(-deliverymonth)

olist <- olist %>% select(-weight_g)
olist_predict <- olist_predict %>% select(-weight_g)

# Remove variable purchaseminute
olist <- olist %>% select(-purchaseminute)
olist_predict <- olist_predict %>% select(-purchaseminute)

# Add dummy variables for categorical variables
olist <- dummy_cols(olist, select_columns = c("purchaseday", "deliveryday", "purchasemonth", "purchasehour", "product_category_major"), remove_first_dummy = TRUE)

olist_predict <- dummy_cols(olist_predict, select_columns = c("purchaseday", "deliveryday", "purchasemonth", "purchasehour", "product_category_major"), remove_first_dummy = TRUE)

# Remove columns for categorical variables for which we just created dummies
olist <- olist %>% select(-c("purchaseday", "deliveryday", "purchasemonth", "purchasehour", "product_category_major"))

olist_predict <- olist_predict %>% select(-c("purchaseday", "deliveryday", "purchasemonth", "purchasehour", "product_category_major"))

############################################################
# Variable Transformation
############################################################

# Determine Skewness of Non-Categorical Variables
skewness_table <- as.data.frame(skewness(olist[,1:8]))

# Perform Log-Transformations of Non-Categorical Variables
olist[,1:8] <- log(olist[,1:8])
olist_predict[,1:7] <- log(olist_predict[,1:7])

# Add Results to Skewness table
skewness_table <- data.frame(skewness_table, skewness(olist[,1:8]))

############################################################
# Train & Test Split
############################################################

# Split data into training and test set 
sample_size <- floor(0.80 * nrow(olist))
training_set <- sample(seq_len(nrow(olist)), size = sample_size, replace = FALSE)

train <- olist[training_set, ]
test <- olist[-training_set, ]

############################################################
# Outlier Removal
############################################################

# Show Boxplot to identify Outliers
boxplot_y_1 <- ggplot(train, aes(y = shippingtime, x = "log(shippingtime)")) +
  geom_boxplot() +
  xlab("") +
  ylab("") +
  ggtitle("Before Outlier Removal")

# Remove Outliers only on the Lower Range
train <- train %>% filter(shippingtime > -1.5)

# Show Boxplot after Outlier Removal
boxplot_y_2 <- ggplot(train, aes(y = shippingtime, x = "log(shippingtime)")) +
  geom_boxplot() +
  xlab("") +
  ylab("") +
  ggtitle("After Outlier Removal")

############################################################
# OLS
############################################################

ols <- lm(shippingtime ~ ., data = train)

fit_ols <- predict(ols, newdata = test)

R2_ols_out <- 1 - (sum((test$shippingtime - fit_ols)^2)/sum((test$shippingtime - mean(test$shippingtime))^2))

MSE_ols_out <- mean((test$shippingtime - fit_ols)^2)

############################################################
# LASSO
############################################################

# Whole lasso path
lasso.cv <- cv.glmnet(as.matrix(train[,c(2:length(train))]), train$shippingtime, 
                      type.measure = "mse", family = "gaussian", nfolds = 10, alpha = 1)

coef_lasso <- coef(lasso.cv, s = "lambda.min") # save for later comparison

# Find number of positive coefficients
length(coef_lasso[coef_lasso[,1] != 0,])

# Calculate lasso R2 and MSE
fit_lasso <- predict(lasso.cv, newx = as.matrix(test[,c(2:length(test))]), 
                          s = lasso.cv$lambda.min)

R2_lasso_out <- 1 - (sum((test$shippingtime - fit_lasso)^2)/sum((test$shippingtime - mean(test$shippingtime))^2))

MSE_lasso_out <- mean((test$shippingtime - fit_lasso)^2)

############################################################
# RIDGE
############################################################

ridge.cv <- cv.glmnet(as.matrix(train[,c(2:length(train))]), train$shippingtime, 
                      type.measure = "mse", family = "gaussian", nfolds = 10, alpha = 0)

coef_ridge <- coef(ridge.cv, s = "lambda.min") # save for later comparison

# Find number of positive coefficients
length(coef_ridge[coef_ridge[,1] != 0,])

# Calculate Ridge R2 and MSE
fit_ridge <- predict(ridge.cv, newx = as.matrix(test[,c(2:length(test))]), 
                          s = ridge.cv$lambda.min)

R2_ridge_out <- 1 - (sum((test$shippingtime - fit_ridge)^2)/sum((test$shippingtime - mean(test$shippingtime))^2))

MSE_ridge_out <- mean((test$shippingtime - fit_ridge)^2)

############################################################
# ELASTIC NET
############################################################

# Create a sequence to search for the optimal alpha between 0 and 1
alphalist <- seq(0,1,by=0.1)

# Create a data frame to store the mean cross-validated error (CVM) for each alpha
alphatable <- data.frame(alpha = alphalist, cv_error = c(NA))

# Write a function that calculates and stores the cross-validated error (CVM) for each alpha
elasticnet <- lapply(alphalist, function(a){
  cv.glmnet(as.matrix(train[,c(2:length(train))]), train$shippingtime, 
                      type.measure = "mse", family = "gaussian", nfolds = 10, alpha = a)
})
for (i in 1:11) {alphatable$cv_error[i] <- min(elasticnet[[i]]$cvm)}

# Find the optimal alpha in our data frame
alphatable[which.min(alphatable$cv_error),]

# Run the Elastic Net with our optimal alpha (= 0.9)
elastic.cv <- cv.glmnet(as.matrix(train[,c(2:length(train))]), train$shippingtime, 
                      type.measure = "mse", family = "gaussian", nfolds = 10, alpha = 0.9)

coef_elastic <- coef(elastic.cv, s = "lambda.min") # save for later comparison

# Find number of positive coefficients
length(coef_elastic[coef_elastic[,1] != 0,])

# Calculate Elastic Net R2 and MSE
fit_elastic <- predict(elastic.cv, newx = as.matrix(test[,c(2:length(test))]), 
                          s = elastic.cv$lambda.min)

R2_elastic_out <- 1 - (sum((test$shippingtime - fit_elastic)^2)/sum((test$shippingtime - mean(test$shippingtime))^2))

MSE_elastic_out <- mean((test$shippingtime - fit_elastic)^2)

############################################################
# FOREST
############################################################

y_olist <- as.matrix(olist[, 1])
x_olist <- as.matrix(olist[, c(2:ncol(olist))])

# Build Forest and automatically tune Parameters
forest <- regression_forest(x_olist[training_set,], 
                            y_olist[training_set,], 
                            tune.parameters = "all")

# Prediction
fit_forest <- predict(forest, newdata = x_olist[-training_set, ])$predictions

# R-squared
SST_forest <- sum(((y_olist[-training_set, ])
