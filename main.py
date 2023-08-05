import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import yfinance as yf
#focus our model on SP500:
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
# clean up our table:
#1. remove unnecessary columns
#2. add a column for a price of a following day and
#   for uprising momentum between each following days AS THE TRAINING TARGET

del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tommorow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tommorow"] > sp500["Close"]).astype(int)

#Training the model with basic parameters of prediction:

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
predictors = ["Close", "Volume", "High", "Low"]
def predict (train, test, predictors, model) :
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

#BackTesting:
#using every 10 previous years (2500 days) to predict the following year (250) by default.
def backtest(data, model, predictors, start= 2500, step=250):
    all_predictions = []
    for i in range (start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

###############################################################
# correct to the days of writing the project, when i test the model accuracy with the current predictors using the lines:
'predictions = backtest(sp500, model, predictors)'
'print(precision_score(predictions["Target", predictions["Predictions"]))'
# i saw a 0.535 acc rate
# meanwhile, the total percentage of uprising momentum from the entire table gave me 0.536
# this indicates that our model falls just behind of thoughtlessly buying at each day opening and selling at it's closing
###############################################################

#Improving the predictors:


#Updating our table -  Moving to a method of ratios instead of absolute prices:
horizons = [2,5,60,250,1000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors+= [ ratio_column, trend_column]
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns!="Tomorrow"])

#Adjusting our training technique based on experiments:
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

#Adjusting our predictor method so it filters days with relatively low probability of success
# (less than 60%)
def predict (train, test, predictors, model) :
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds>=6] = 1
    preds[preds<0.6] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

####################################################################
'Now, using the same test i recieve a success rate of near 0.58 which comes to show'
'that our model WOULD BE BENEFICIAL for years 1990 - 2022'

####################################################################
