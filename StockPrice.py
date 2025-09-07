"""
Author : Gouhar khalid harris 
Date : jan 2024
"""
import yfinance as yf
import sklearn as sk
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
pd.options.display.max_columns = None
import yfinance as yf
#load data of Apple index
d=yf.Ticker("AAPL")
dt=d.history(period="max")
print(dt)

#cleaning data
del dt["Dividends"]
del dt["Stock Splits"]
dt=dt.loc["1990-01-01":].copy()
#setting target values
dt["nxt"]=dt["Close"].shift(-1)
dt["tom"]=(dt["nxt"]>=dt["Close"]).astype(int)


#dataset spliting for test and train
xtr=dt.iloc[:-100]
xte=dt.iloc[-100:]


#training the classifier
lrn=RandomForestClassifier(n_estimators=90,min_samples_split=150, random_state=1)
pr=["Close","Volume","Open","High","Low"]
lrn.fit(xtr[pr],xtr["tom"])
#predicting
pred=lrn.predict(xte[pr])
pred=pd.Series(pred,index=xte.index)



#evaluation of the model
score=precision_score(xte["tom"],pred)
print("precision score :",score)
com=pd.concat([xte["tom"],pred],axis=1)
com.plot()
pl.show()
