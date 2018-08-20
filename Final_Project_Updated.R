rm(list = ls())
library(plyr)
library(dplyr)
library(lubridate)
library(randomForest)
library(caret)
library(ggplot2)
library(nnet)
library(ROCR)
library(DMwR)

# Function to determine season based on date supplied
getSeason <- function(DATES) {
  WS <- as.Date("2012-12-21", format = "%Y-%m-%d") # Winter Solstice
  SE <- as.Date("2012-3-20",  format = "%Y-%m-%d") # Spring Equinox
  SS <- as.Date("2012-6-21",  format = "%Y-%m-%d") # Summer Solstice
  FE <- as.Date("2012-9-22",  format = "%Y-%m-%d") # Fall Equinox
  
  # Convert dates from any year to 2017 dates
  d <- as.Date(strftime(DATES, format="%2012-%m-%d"))
  
  ifelse (d >= WS | d < SE, "Winter",
          ifelse (d >= SE & d < SS, "Spring",
                  ifelse (d >= SS & d < FE, "Summer", "Fall")))
}

# Function to create the Air Qualtiy Index (AQI)
aqi <- function(d.in) {
  
  O3_index <- NA
  PM10_index <- NA
  PM25_index <- NA
  NO2_index <- NA
  
  if(is.na(d.in$O3)){
    O3_index <- 0
  }
  else if (d.in$O3>100){
    O3_index <- 4
  }
  else if (d.in$O3 >= 67){
    O3_index <- 3
  }
  else if (d.in$O3 >= 34){
    O3_index <- 2
  }
  else{
    O3_index <- 1
  }
  if(is.na(d.in$PM10)){
    PM10_index <- 0
  }
  else if (d.in$PM10 >= 59){
    PM10_index <- 5
  }
  else if (d.in$PM10 >= 51){
    PM10_index <- 4
  }
  else if (d.in$PM10 >= 34){
    PM10_index <- 3
  }
  else if (d.in$PM10 >= 17){
    PM10_index <- 2
  }
  else{
    PM10_index <- 1
  }
  if (is.na(d.in$PM25)){
    PM25_index <- 0
  }
  else if (d.in$PM25 >= 48){
    PM25_index <- 6
  }
  else if (d.in$PM25 >= 42){
    PM25_index <- 5
  }
  else if (d.in$PM25 >= 36){
    PM25_index <- 4
  }
  else if (d.in$PM25 >= 24){
    PM25_index <- 3
  }
  else if (d.in$PM25 >= 12){
    PM25_index <- 2
  }
  else{
    PM25_index <- 1
  }
  if (is.na(d.in$NO2)){
    NO2_index <- 0
  }
  else if (d.in$NO2 >= 68){
    NO2_index <- 2
  }
  else{
    NO2_index <- 1
  }
  
  result <- max(O3_index, PM10_index, PM25_index, NO2_index)
  
  return(result)
  
}

d.in <- read.csv("~/Downloads/train.csv", header = TRUE)

regions <- read.csv("~/Downloads/regions.csv", header = TRUE)

summary(d.in)

# Data wrangling
d.in <- d.in %>% mutate(Id = as.factor(Id),
                        region = as.factor(region),
                        date = ymd(date),
                        mortality_rate = as.numeric(mortality_rate),
                        O3 = as.numeric(O3),
                        PM10 = as.numeric(PM10),
                        PM25 = as.numeric(PM25),
                        NO2 = as.numeric(NO2),
                        T2M = as.numeric(T2M))

# Let's get the Mean and SD for each of our base predictors
O3.avg <- d.in %>% select(O3) %>% filter(!is.na(.)) %>% unlist() %>% mean
O3.sd <- d.in %>% select(O3) %>% filter(!is.na(.)) %>% unlist() %>% sd

NO2.avg <- d.in %>% select(NO2) %>% filter(!is.na(.)) %>% unlist() %>% mean
NO2.sd <- d.in %>% select(NO2) %>% filter(!is.na(.)) %>% unlist() %>% sd

PM10.avg <- d.in %>% select(PM10) %>% filter(!is.na(.)) %>% unlist() %>% mean
PM10.sd <- d.in %>% select(PM10) %>% filter(!is.na(.)) %>% unlist() %>% sd

PM25.avg <- d.in %>% select(PM25) %>% filter(!is.na(.)) %>% unlist() %>% mean
PM25.sd <- d.in %>% select(PM25) %>% filter(!is.na(.)) %>% unlist() %>% sd

T2M.avg <- d.in %>% select(T2M) %>% filter(!is.na(.)) %>% unlist() %>% mean
T2M.sd <- d.in %>% select(T2M) %>% filter(!is.na(.)) %>% unlist() %>% sd

# Determine quartiles for each base predictor to determine categorey splits for feature engineering.
# Adapative Binning purposes.
O3.qaurtile <- quantile(eval(parse(text = "d.in %>% select(O3) %>% filter(!is.na(.)) %>% unlist()")))
NO2.quartile <- quantile(eval(parse(text = "d.in %>% select(NO2) %>% filter(!is.na(.)) %>% unlist()")))
PM10.quartile <- quantile(eval(parse(text = "d.in %>% select(PM10) %>% filter(!is.na(.)) %>% unlist()")))
PM25.quartile <- quantile(eval(parse(text = "d.in %>% select(PM25) %>% filter(!is.na(.)) %>% unlist()")))
T2M.quartile <- quantile(eval(parse(text = "d.in %>% select(T2M) %>% filter(!is.na(.)) %>% unlist()")))

# Dataset with imputed values for NAs using kNN imputation approach
# We can't scale categorical variables. No date, No id(there are many), No response variable
d.in.imputed <- knnImputation(d.in[, !names(d.in) %in% c("Id", "date", "mortality_rate")], k=7)
d.in.two <- d.in[, names(d.in) %in% c("Id", "date", "mortality_rate")] 
d.in.imputed <- bind_cols(d.in.two, d.in.imputed)

# Feature Engineering on d.in (non-imputed dataframe): Adaptive Binning - building categorical features for each predictor based on predictor quartiles
d.in$O3_Level[d.in$O3 > 55.88100] <- "O3_High"
d.in$O3_Level[d.in$O3 > 35.07425 & d.in$O3 <= 55.88100] <- "O3_Moderate"
d.in$O3_Level[d.in$O3 <= 35.07425] <- "O3_Low"

d.in$NO2_Level[d.in$NO2 > 15.858] <- "NO2_High"
d.in$NO2_Level[d.in$NO2 > 6.056 & d.in$NO2 <= 15.858] <- "NO2_Moderate"
d.in$NO2_Level[d.in$NO2 <= 6.056] <- "NO2_Low"

d.in$PM10_Level[d.in$PM10 > 16.58900] <- "PM10_High"
d.in$PM10_Level[d.in$PM10 > 8.65625 & d.in$PM10 <= 16.58900] <- "PM10_Moderate"
d.in$PM10_Level[d.in$PM10 <= 8.65625] <- "PM10_Low"

d.in$PM25_Level[d.in$PM25 > 9.3265] <- "PM25_High"
d.in$PM25_Level[d.in$PM25 > 3.6240 & d.in$PM25 <= 9.3265] <- "PM25_Moderate"
d.in$PM25_Level[d.in$PM25 <= 3.6240] <- "PM25_Low"

d.in$T2M_Level[d.in$T2M > 287.2405] <- "T2M_High"
d.in$T2M_Level[d.in$T2M > 279.3215 & d.in$T2M <= 287.2405] <- "T2M_Moderate"
d.in$T2M_Level[d.in$T2M <= 279.3215] <- "T2M_Low"

# Feature engineer a Season variable
d.in$Season <- sapply(d.in$date, function(x) getSeason(x))

# Feature engineer an AQI index variable
for(i in 1:nrow(d.in)) 
{
  d.in[i,16] <- aqi(d.in[i,])
}

names(d.in)[names(d.in) == 'V16'] <- 'AQI'

# Ensure factor variables are as such in the dataset
d.in <- d.in %>% mutate(O3_Level = as.factor(O3_Level),
                        NO2_Level = as.factor(NO2_Level),
                        PM10_Level = as.factor(PM10_Level),
                        PM25_Level = as.factor(PM25_Level),
                        T2M_Level = as.factor(T2M_Level),
                        Season = as.factor(Season),
                        AQI = as.factor(AQI))
# --------------------------------------------------------------------------------------
# Feature Engineering on d.in.imputed (imputed dataframe): Adaptive Binning - building categorical features for each predictor based on predictor quartiles
d.in.imputed$O3_Level[d.in.imputed$O3 > 55.88100] <- "O3_High"
d.in.imputed$O3_Level[d.in.imputed$O3 > 35.07425 & d.in.imputed$O3 <= 55.88100] <- "O3_Moderate"
d.in.imputed$O3_Level[d.in.imputed$O3 <= 35.07425] <- "O3_Low"

d.in.imputed$NO2_Level[d.in.imputed$NO2 > 15.858] <- "NO2_High"
d.in.imputed$NO2_Level[d.in.imputed$NO2 > 6.056 & d.in.imputed$NO2 <= 15.858] <- "NO2_Moderate"
d.in.imputed$NO2_Level[d.in.imputed$NO2 <= 6.056] <- "NO2_Low"

d.in.imputed$PM10_Level[d.in.imputed$PM10 > 16.58900] <- "PM10_High"
d.in.imputed$PM10_Level[d.in.imputed$PM10 > 8.65625 & d.in.imputed$PM10 <= 16.58900] <- "PM10_Moderate"
d.in.imputed$PM10_Level[d.in.imputed$PM10 <= 8.65625] <- "PM10_Low"

d.in.imputed$PM25_Level[d.in.imputed$PM25 > 9.3265] <- "PM25_High"
d.in.imputed$PM25_Level[d.in.imputed$PM25 > 3.6240 & d.in.imputed$PM25 <= 9.3265] <- "PM25_Moderate"
d.in.imputed$PM25_Level[d.in.imputed$PM25 <= 3.6240] <- "PM25_Low"

d.in.imputed$T2M_Level[d.in.imputed$T2M > 287.2405] <- "T2M_High"
d.in.imputed$T2M_Level[d.in.imputed$T2M > 279.3215 & d.in.imputed$T2M <= 287.2405] <- "T2M_Moderate"
d.in.imputed$T2M_Level[d.in.imputed$T2M <= 279.3215] <- "T2M_Low"

# Feature engineer a Season variable
d.in.imputed$Season <- sapply(d.in.imputed$date, function(x) getSeason(x))

# Feature engineer an AQI index variable
for(i in 1:nrow(d.in.imputed)) 
{
  d.in.imputed[i,16] <- aqi(d.in.imputed[i,])
}

names(d.in.imputed)[names(d.in.imputed) == 'V16'] <- 'AQI'

# Ensure factor variables are as such in the dataset
d.in.imputed <- d.in.imputed %>% mutate(O3_Level = as.factor(O3_Level),
                                        NO2_Level = as.factor(NO2_Level),
                                        PM10_Level = as.factor(PM10_Level),
                                        PM25_Level = as.factor(PM25_Level),
                                        T2M_Level = as.factor(T2M_Level),
                                        Season = as.factor(Season),
                                        AQI = as.factor(AQI))
# --------------------------------------------------------------------------------------
# Dataset with all NAs removed - for the first round of modeling
d.in.complete <- d.in[complete.cases(d.in),]
# --------------------------------------------------------------------------------------
# Engineering binary features for each category for each categorical variable so we can use the categorical variables in our neural network
Region.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$region), function(x) {(d.in.complete$region == x)*1})))
names(Region.flags.complete) = levels(d.in.complete$region)
d.in.complete = cbind(d.in.complete, Region.flags.complete)

O3.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$O3_Level), function(x) {(d.in.complete$O3_Level == x)*1})))
names(O3.flags.complete) = levels(d.in.complete$O3_Level)
d.in.complete = cbind(d.in.complete, O3.flags.complete)

NO2.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$NO2_Level), function(x) {(d.in.complete$NO2_Level == x)*1})))
names(NO2.flags.complete) = levels(d.in.complete$NO2_Level)
d.in.complete = cbind(d.in.complete, NO2.flags.complete)

PM10.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$PM10_Level), function(x) {(d.in.complete$PM10_Level == x)*1})))
names(PM10.flags.complete) = levels(d.in.complete$PM10_Level)
d.in.complete = cbind(d.in.complete, PM10.flags.complete)

PM25.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$PM25_Level), function(x) {(d.in.complete$PM25_Level == x)*1})))
names(PM25.flags.complete) = levels(d.in.complete$PM25_Level)
d.in.complete = cbind(d.in.complete, PM25.flags.complete)

T2M.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$T2M_Level), function(x) {(d.in.complete$T2M_Level == x)*1})))
names(T2M.flags.complete) = levels(d.in.complete$T2M_Level)
d.in.complete = cbind(d.in.complete, T2M.flags.complete)

Season.flags.complete <- data.frame(Reduce(cbind, lapply(levels(d.in.complete$Season), function(x) {(d.in.complete$Season == x)*1})))
names(Season.flags.complete) = levels(d.in.complete$Season)
d.in.complete = cbind(d.in.complete, Season.flags.complete)

Region.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$region), function(x) {(d.in.imputed$region == x)*1})))
names(Region.flags.imputed) = levels(d.in.imputed$region)
d.in.imputed = cbind(d.in.imputed, Region.flags.imputed)

O3.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$O3_Level), function(x) {(d.in.imputed$O3_Level == x)*1})))
names(O3.flags.imputed) = levels(d.in.imputed$O3_Level)
d.in.imputed = cbind(d.in.imputed, O3.flags.imputed)

NO2.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$NO2_Level), function(x) {(d.in.imputed$NO2_Level == x)*1})))
names(NO2.flags.imputed) = levels(d.in.imputed$NO2_Level)
d.in.imputed = cbind(d.in.imputed, NO2.flags.imputed)

PM10.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$PM10_Level), function(x) {(d.in.imputed$PM10_Level == x)*1})))
names(PM10.flags.imputed) = levels(d.in.imputed$PM10_Level)
d.in.imputed = cbind(d.in.imputed, PM10.flags.imputed)

PM25.flags.imputed<- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$PM25_Level), function(x) {(d.in.imputed$PM25_Level == x)*1})))
names(PM25.flags.imputed) = levels(d.in.imputed$PM25_Level)
d.in.imputed = cbind(d.in.imputed, PM25.flags.imputed)

T2M.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$T2M_Level), function(x) {(d.in.imputed$T2M_Level == x)*1})))
names(T2M.flags.imputed) = levels(d.in.imputed$T2M_Level)
d.in.imputed = cbind(d.in.imputed, T2M.flags.imputed)

Season.flags.imputed <- data.frame(Reduce(cbind, lapply(levels(d.in.imputed$Season), function(x) {(d.in.imputed$Season == x)*1})))
names(Season.flags.imputed) = levels(d.in.imputed$Season)
d.in.imputed = cbind(d.in.imputed, Season.flags.imputed)

rm(d.in.two, O3.flags.complete, NO2.flags.complete, PM10.flags.complete, PM25.flags.complete, T2M.flags.complete, O3.flags.imputed, NO2.flags.imputed, 
   PM10.flags.imputed, PM25.flags.imputed, T2M.flags.imputed, Season.flags.imputed, Season.flags.complete, Region.flags.complete, Region.flags.imputed)
# --------------------------------------------------------------------------------------
### ANN ###
# 1. Split data into Training and Test
#     a. 70/30
#     b. 80/20
# 2. ANN
#     a. internal network size: 2 - 25 w/ constant decay
# 3. Repeat steps 1. and 2. for the imputed dataset

# Running ANN on removed incomplete records data set

calculate_rmse <- function(d){
  # function to calculate rmse
  rmse <- sqrt(mean((d$predicted_rate - d$actual_rate)^2))
  return(rmse) 
}

# Normalizing the dataset so values fall between [0, 1]
scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

# Scale the data in the d.in.complete data set
d.in.complete.scaled <- d.in.complete %>% 
  select(O3, NO2, PM10, PM25, T2M, mortality_rate) %>%
  mutate_all(scale01)

# d.in.complete.scaled <- as.data.frame(scale(d.in.complete[c("O3", "NO2", "PM10", "PM25", "T2M", "mortality_rate")]))
d.col.diff <- d.in.complete[, !names(d.in.complete) %in% c("O3", "NO2", "PM10", "PM25", "T2M", "mortality_rate")]
d.in.complete.scaled <- bind_cols(d.col.diff, d.in.complete.scaled)

# Scale the data in the d.in.imputed data set
d.in.imputed.scaled <- d.in.imputed %>% 
  select(O3, NO2, PM10, PM25, T2M, mortality_rate) %>%
  mutate_all(scale01)

# d.in.complete.scaled <- as.data.frame(scale(d.in.complete[c("O3", "NO2", "PM10", "PM25", "T2M", "mortality_rate")]))
d.col.diff <- d.in.imputed[, !names(d.in.imputed) %in% c("O3", "NO2", "PM10", "PM25", "T2M", "mortality_rate")]
d.in.imputed.scaled <- bind_cols(d.col.diff, d.in.imputed.scaled)

# Creating the 70/30 Train/Test split on d.in.complete
set.seed(123)

train_index_70_30_complete <- createDataPartition(d.in.complete.scaled$mortality_rate, p = .7, list = FALSE, times = 1)
train_index_70_30_imputed <- createDataPartition(d.in.imputed.scaled$mortality_rate, p = .7, list = FALSE, times = 1)

d.train.complete.70.30 <- d.in.complete.scaled[train_index_70_30_complete,]
d.test.complete.70.30 <- d.in.complete.scaled[-train_index_70_30_complete,]

d.train.imputed.70.30 <- d.in.imputed.scaled[train_index_70_30_imputed,]
d.test.imputed.70.30 <- d.in.imputed.scaled[-train_index_70_30_imputed,]

# Due to computing resource limitations on the development machine, tuning parameters will not be as comprehensive.
fitControl <- trainControl(method = "cv", number = 5) # try number = 5, 7
nnetGrid <- expand.grid(size = seq(from = 2, to = 25, by = 1),
                        decay = c(5e-4)) # decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

ann.1 <- train(mortality_rate ~ O3 + NO2 + PM10 + PM25 + T2M + AQI + E12000001 + E12000002 + E12000003 + E12000004 + E12000005 + E12000006 + E12000007 + E12000008
               + E12000009 + O3_High + O3_Moderate + O3_Low + NO2_High + NO2_Moderate + NO2_Low + PM10_High + PM10_Moderate + PM10_Low + PM25_High + PM25_Moderate
               + PM25_Low + T2M_High + T2M_Moderate + T2M_Low + Fall + Spring + Summer + Winter,
               data = d.train.complete.70.30,
               method = "nnet",
               trControl = fitControl,
               tuneGrid = nnetGrid,
               trace = FALSE,
               maxit = 200,
               linout = 1)

ann.2 <- train(mortality_rate ~ O3 + NO2 + PM10 + PM25 + T2M + AQI + E12000001 + E12000002 + E12000003 + E12000004 + E12000005 + E12000006 + E12000007 + E12000008
               + E12000009 + O3_High + O3_Moderate + O3_Low + NO2_High + NO2_Moderate + NO2_Low + PM10_High + PM10_Moderate + PM10_Low + PM25_High + PM25_Moderate
               + PM25_Low + T2M_High + T2M_Moderate + T2M_Low + Fall + Spring + Summer + Winter,
               data = d.train.imputed.70.30,
               method = "nnet",
               trControl = fitControl,
               tuneGrid = nnetGrid,
               trace = FALSE,
               maxit = 200,
               linout = 1)

d.rmse <- data.frame(Model = as.character(),
                     Train_Test_Split = as.character(),
                     Data_Set_Preprocessing = as.character(),
                     RMSE = as.numeric(),
                     Accuracy = as.numeric())

d.temp.complete.7030 <- d.test.complete.70.30
d.temp.complete.7030$pred_mortality_rate <- predict(ann.1, newdata = d.temp.complete.7030)
d.temp.complete.7030$actual_rate <- d.temp.complete.7030$mortality_rate
d.temp.complete.7030$predicted_rate <- d.temp.complete.7030$pred_mortality_rate
h.rmse.complete.7030 <- calculate_rmse(d.temp.complete.7030)

d.temp.imputed.7030 <- d.test.imputed.70.30
d.temp.imputed.7030$pred_mortality_rate <- predict(ann.2, newdata = d.temp.imputed.7030)
d.temp.imputed.7030$actual_rate <- d.temp.imputed.7030$mortality_rate
d.temp.imputed.7030$predicted_rate <- d.temp.imputed.7030$pred_mortality_rate
h.rmse.imputed.7030 <- calculate_rmse(d.temp.imputed.7030)

h.model.complete.7030 <- "NN1"
h.model.imputed.7030 <- "NN2"
h.split.7030 <- "70-30"
h.dataSet.complete.7030 <- "NA Recoreds Omitted"
h.dataSet.imputed.7030 <- "kNN Imputation"

acc.complete.7030 <- cor(d.temp.complete.7030$actual_rate, d.temp.complete.7030$predicted_rate)
acc.imputed.7030 <- cor(d.temp.imputed.7030$actual_rate, d.temp.imputed.7030$predicted_rate)

# store output
d.rmse <- rbind(d.rmse, data.frame(Model = as.character(h.model.complete.7030), Train_Test_Split = as.character(h.split.7030), Data_Set_Preprocessing = as.character(h.dataSet.complete.7030), RMSE = as.numeric(h.rmse.complete.7030), Accuracy = as.numeric(acc.complete.7030)))
d.rmse <- rbind(d.rmse, data.frame(Model = as.character(h.model.imputed.7030), Train_Test_Split = as.character(h.split.7030), Data_Set_Preprocessing = as.character(h.dataSet.imputed.7030), RMSE = as.numeric(h.rmse.imputed.7030), Accuracy = as.numeric(acc.imputed.7030)))

# rm(d.temp.complete.7030, d.temp.imputed.7030, d.test.imputed.70.30, d.test.complete.70.30, d.train.imputed.70.30, d.train.complete.70.30)
# --------------------------------------------------------------------------------------
# Creating the 80/20 Train/Test split on d.in.complete
set.seed(234)

train_index_80_20_complete <- createDataPartition(d.in.complete.scaled$mortality_rate, p = .8, list = FALSE, times = 1)
train_index_80_20_imputed <- createDataPartition(d.in.imputed.scaled$mortality_rate, p = .8, list = FALSE, times = 1)

d.train.complete.80.20 <- d.in.complete.scaled[train_index_80_20_complete,]
d.test.complete.80.20 <- d.in.complete.scaled[-train_index_80_20_complete,]

d.train.imputed.80.20 <- d.in.imputed.scaled[train_index_80_20_imputed,]
d.test.imputed.80.20 <- d.in.imputed.scaled[-train_index_80_20_imputed,]

ann.3 <- train(mortality_rate ~ O3 + NO2 + PM10 + PM25 + T2M + AQI + E12000001 + E12000002 + E12000003 + E12000004 + E12000005 + E12000006 + E12000007 + E12000008
               + E12000009 + O3_High + O3_Moderate + O3_Low + NO2_High + NO2_Moderate + NO2_Low + PM10_High + PM10_Moderate + PM10_Low + PM25_High + PM25_Moderate
               + PM25_Low + T2M_High + T2M_Moderate + T2M_Low + Fall + Spring + Summer + Winter,
               data = d.train.complete.80.20,
               method = "nnet",
               trControl = fitControl,
               tuneGrid = nnetGrid,
               trace = FALSE,
               maxit = 200,
               linout = 1)

ann.4 <- train(mortality_rate ~ O3 + NO2 + PM10 + PM25 + T2M + AQI + E12000001 + E12000002 + E12000003 + E12000004 + E12000005 + E12000006 + E12000007 + E12000008
               + E12000009 + O3_High + O3_Moderate + O3_Low + NO2_High + NO2_Moderate + NO2_Low + PM10_High + PM10_Moderate + PM10_Low + PM25_High + PM25_Moderate
               + PM25_Low + T2M_High + T2M_Moderate + T2M_Low + Fall + Spring + Summer + Winter,
               data = d.train.imputed.80.20,
               method = "nnet",
               trControl = fitControl,
               tuneGrid = nnetGrid,
               trace = FALSE,
               maxit = 200,
               linout = 1)

d.temp.complete.8020 <- d.test.complete.80.20
d.temp.complete.8020$pred_mortality_rate <- predict(ann.3, newdata = d.temp.complete.8020)
d.temp.complete.8020$actual_rate <- d.temp.complete.8020$mortality_rate
d.temp.complete.8020$predicted_rate <- d.temp.complete.8020$pred_mortality_rate
h.rmse.complete.8020 <- calculate_rmse(d.temp.complete.8020)

d.temp.imputed.8020 <- d.test.imputed.80.20
d.temp.imputed.8020$pred_mortality_rate <- predict(ann.4, newdata = d.temp.imputed.8020)
d.temp.imputed.8020$actual_rate <- d.temp.imputed.8020$mortality_rate
d.temp.imputed.8020$predicted_rate <- d.temp.imputed.8020$pred_mortality_rate
h.rmse.imputed.8020 <- calculate_rmse(d.temp.imputed.8020)

h.model.complete.8020 <- "NN3"
h.model.imputed.8020 <- "NN4"
h.split.8020 <- "80-20"
h.dataSet.complete.8020 <- "NA Recoreds Omitted"
h.dataSet.imputed.8020 <- "kNN Imputation"

acc.complete.8020 <- cor(d.temp.complete.8020$actual_rate, d.temp.complete.8020$predicted_rate)
acc.imputed.8020 <- cor(d.temp.imputed.8020$actual_rate, d.temp.imputed.8020$predicted_rate)

# store output
d.rmse <- rbind(d.rmse, data.frame(Model = as.character(h.model.complete.8020), Train_Test_Split = as.character(h.split.8020), Data_Set_Preprocessing = as.character(h.dataSet.complete.8020), RMSE = as.numeric(h.rmse.complete.8020), Accuracy = as.numeric(acc.complete.8020)))
d.rmse <- rbind(d.rmse, data.frame(Model = as.character(h.model.imputed.8020), Train_Test_Split = as.character(h.split.8020), Data_Set_Preprocessing = as.character(h.dataSet.imputed.8020), RMSE = as.numeric(h.rmse.imputed.8020), Accuracy = as.numeric(acc.imputed.8020)))

# rm(d.temp.complete.8020, d.temp.imputed.8020, d.test.imputed.80.20, d.test.complete.80.20, d.train.imputed.80.20, d.train.complete.80.20)
# --------------------------------------------------------------------------------------
p <- ggplot(d.rmse, aes(y = Accuracy, x = Model)) + geom_col() + ggtitle("Model Performance") + theme_classic()

plot(p)
# --------------------------------------------------------------------------------------

# Model         |     Train/Test Data Split     |     Data PreProcessing     |     Model Accuracy  
# --------------|-------------------------------|----------------------------|--------------------
# LM1           |             70-30             |     NA Recoreds Omitted    |      0.5959228     
# LM2           |             70-30             |     kNN Imputation         |      0.5731310
# LM3           |             80-20             |     NA Recoreds Omitted    |      0.5966913
# LM4           |             80-20             |     kNN Imputation         |      0.5619385
# RF1           |             70-30             |     NA Recoreds Omitted    |      0.7704970
# RF2           |             70-30             |     kNN Imputation         |      0.7572889
# RF3           |             80-20             |     NA Recoreds Omitted    |      0.7639935
# RF4           |             80-20             |     kNN Imputation         |      0.7485153
# NN1           |             70-30             |     NA Recoreds Omitted    |      0.7673420
# NN2           |             70-30             |     kNN Imputation         |      0.7539171
# NN3           |             80-20             |     NA Recoreds Omitted    |      0.7759289
# NN4           |             80-20             |     kNN Imputation         |      0.7455035 

d.results <- data.frame(Model = as.character(),
                     Train_Test_Split = as.character(),
                     Data_Set_Preprocessing = as.character(),
                     Accuracy = as.numeric())

d.results <- rbind(d.results, data.frame(Model = as.character("LM1"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.5959228)))
d.results <- rbind(d.results, data.frame(Model = as.character("LM2"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.5731310)))
d.results <- rbind(d.results, data.frame(Model = as.character("LM3"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.5966913)))
d.results <- rbind(d.results, data.frame(Model = as.character("LM4"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.5619385)))
d.results <- rbind(d.results, data.frame(Model = as.character("RF1"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.7704970)))
d.results <- rbind(d.results, data.frame(Model = as.character("RF2"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.7572889)))
d.results <- rbind(d.results, data.frame(Model = as.character("RF3"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.7639935)))
d.results <- rbind(d.results, data.frame(Model = as.character("RF4"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.7485153)))
d.results <- rbind(d.results, data.frame(Model = as.character("NN1"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.7673420)))
d.results <- rbind(d.results, data.frame(Model = as.character("NN2"), Train_Test_Split = as.character("70-30"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.7539171)))
d.results <- rbind(d.results, data.frame(Model = as.character("NN3"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("NA Recoreds Omitted"), Accuracy = as.numeric(0.7759289)))
d.results <- rbind(d.results, data.frame(Model = as.character("NN4"), Train_Test_Split = as.character("80-20"), Data_Set_Preprocessing = as.character("kNN Imputation"), Accuracy = as.numeric(0.7455035)))

p <- ggplot(d.results, aes(y = Accuracy, x = Model, fill = Train_Test_Split)) + geom_col() + ggtitle("Model Performance") + theme_classic()

plot(p)