# S&P 500 Machine Learning Project
The following project was made by me to showcase the knowledge i've acquired in the field of Machine Learning. The project is about creating a model to predict whether the price of the S&P 500 index will increase or decrease based on historical data. It includes the process of examining various models and selecting one accordingly, training it, backtesting, and optimizing it's prediction tools&methods, aswell as cleaning the data fed to it. Below listed further information regarding each phase.

## Table of Contents
- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Epilogue](#epilogue)

## Introduction
In this project, we utilize historical stock price data for the S&P 500 index and four major technology companies: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), and Amazon (AMZN). Our aim is to build a predictive model that forecasts whether the S&P 500 index will experience a price increase or decrease in the following day based on a set of features derived from historical price and volume data.

## Data Collection
We use the `yfinance` library to collect historical stock price data for the S&P 500 index. If the data is not locally available, it is fetched from the web and stored as a CSV file for future use.

## Data Preprocessing
The collected data is cleaned and preprocessed. Unnecessary columns such as dividends and stock splits are removed. The data is then divided into training and test sets.

## Feature Engineering
We explore various predictors to enhance our model's accuracy. These include:
- Creating rolling averages and ratios based on historical stock price data.
- Calculating trend indicators to capture historical price movements.

## Model Selection
We evaluate three machine learning models: Logistic Regression, Random Forest Classifier, and Support Vector Classifier. Model performance is assessed using precision scores and ranked to identify the most suitable model.

## Model Evaluation
The chosen model, Random Forest Classifier, is further optimized by tuning hyperparameters and utilizing rolling average-based predictors. The model's accuracy, precision, recall, and F1-score are evaluated and compared for different prediction horizons.

## Visualization
To visualize model evaluation metrics, we utilize Seaborn to create bar plots illustrating the accuracy, precision, recall, and F1-score of each model.

## Epilogue
We explore the correlation between the S&P 500 index and the top four technology stocks. Additionally, we visualize the rate of return for these stocks over the past five years, providing insights into their performance and correlation with the S&P 500 index.

This project serves as a comprehensive demonstration of machine learning techniques applied to financial data, showcasing data preprocessing, feature engineering, model selection, evaluation, and visualization.

