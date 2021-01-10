import yfinance as yf
import pandas as pd
import requests
from time import sleep
from config import *
from datetime import datetime, date, timedelta
import numpy as np
import os

holdings = open('data/qqq.csv').readlines()

symbols = [holding.split(',')[2].strip() for holding in holdings][1:]

def get_historic_data(startDate, endDate):
    for symbol in symbols:
        data = yf.download(symbol, startDate, endDate)
        #print(data)
        data.to_csv("data/prices/{}.csv".format(symbol))

def update_price_data():
    tom = date.today() + timedelta(days=1)
    min_time = datetime.min.time()
    dateTime = datetime.combine(tom, min_time)
    endDate = dateTime.strftime("%Y-%m-%d")

    updatedSymbols = []

    for symbol in symbols:
        oldData = pd.read_csv("data/prices/{}.csv".format(symbol))
        startDate = oldData["Date"][0]
        oldEndDate = oldData["Date"].iloc[-1]
        newData = yf.download(symbol, startDate, endDate)
        newEndDate = newData.index[-1].to_pydatetime()
        newEndDate = newEndDate.strftime("%Y-%m-%d")
        if (newEndDate != oldEndDate):
            newData.to_csv("data/prices/{}.csv".format(symbol))
            updatedSymbols.append(symbol)
        else:
            print("{} has not updated".format(symbol))
    

    return updatedSymbols


def get_latest_price(symbol):
    data = pd.read_csv("data/historicalPrices/{}.csv".format(symbol))
    
    print("getting reponse for {}".format(symbol))
    r = requests.get('https://finnhub.io/api/v1/quote?symbol={}&token={}'.format(symbol, FINNHUB_API_KEY))
    closePrice = r.json().get("c")
    highPrice = r.json().get("h")
    lowPrice = r.json().get("l")
    openPrice = r.json().get("o")
    timePrice = r.json().get("t")
    date = datetime.fromtimestamp(timePrice).strftime('%Y-%m-%d')
    #print(r.json())
    print((data["Date"].iloc[-1]))
    #print(date)
    # print(date == data["Date"].iloc[-1])

    #print(type(data["Date"]))
    if(not data['Date'].str.contains(date).any()):
        new_row = {'Date': date, 'Open': openPrice, 'High': highPrice, 'Low': lowPrice, 'Close': closePrice, 'Adj Close': np.nan, "Volume": np.nan}
        data = data.append(new_row, ignore_index=True)
    else:
        print("Date already exists")

    data.index = data['Date'] 
    del data['Date'] 

    data.to_csv("data/historicalPrices/{}.csv".format(symbol))
    sleep(2)

#get_historic_data('2020-01-01','2020-12-31')

# for symbol in symbols:
#     get_latest_price(symbol)

# update_price_data()