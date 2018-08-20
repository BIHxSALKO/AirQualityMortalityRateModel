# AirQualityMortalityRateModel
Predictive modeling on mortality rates in England based on Air Quality Indicators.

Data was acquired from Kaggle. Data wrangling, feature selection and feature engineering techniques were used to finalize
the data for modeling purposes

Throughout this project we explore multiple predictive models to generate a forcasted mortality rate value. We explored
1. Multiple Linear Regressions
2. Random Forests
3. Neural Networks

We explored two train/test splits: 70/30 and 80/20, and for each split we used record removal and knn imputation for missing 
value treatment. Additionally, for the random forests and neural networks we explored small hyper parameter grids as local 
computation resources were limited.

Accuracy Achieved: 0.7759
