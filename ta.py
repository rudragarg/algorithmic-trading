import btalib
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import yfinance as yf
import datetime as dt
import datetime as dt

import nltk
#nltk.download('vader_lexicon')
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
finwiz_url = 'https://finviz.com/quote.ashx?t='

def get_news_sentiment():
    news_tables = {}

    holdings = open('data/qqq.csv').readlines()

    tickers = [holding.split(',')[2].strip() for holding in holdings][1:]

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, "lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table
    
    
    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text() 
            # splite text in the td tag into a list 
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                time = date_scrape[0]
                
            # else load 'date' as the 1st element and 'time' as the second    
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'  
            ticker = file_name.split('_')[0]
            
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, text])
    
            
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

    parsed_and_scored_news.head()

    tempMin = parsed_and_scored_news.groupby(["ticker", "date"]).compound.min()
    tempMax = parsed_and_scored_news.groupby(["ticker", "date"]).compound.max()


    result = tempMax + tempMin

    return result

def get_data(symbol):
    return yf.download(symbol,'2017-01-01','2020-12-31')

def get_entry_price(data):
    return data.iloc[0,3]

def get_exit_profit(data):
    return data.iloc[-1, 3] - data.iloc[0,3]

def get_BBands(data):
    mid, top, bot = btalib.bbands(data, period = 20, devs = 2.0)
    midSell, topSell, botSell = btalib.bbands(data, period = 20, devs = 2.0)


    data["Mid BBand"] = list(mid)
    data["Top BBand"] = list(top)
    data["Bot BBand"] = list(bot)
    data["Sell BBand"] = list(botSell)

    # BBValue = (data["Close"] - data["Mid BBand"])/ (2 * data["Close"].std(ddof=0))
    # data["BB Value"] = BBValue

    return data

def get_EMA(data):
    ShortEMA = btalib.ema(data, period=12)
    LongEMA = btalib.ema(data, period=26)

    data["ShortEMA"] = ShortEMA.df
    data["LongEMA"] = LongEMA.df

    return data

def get_RSI(data):
    rsi = btalib.rsi(data, period = 14)

    data["RSI"] = rsi.df

    return data

#Implementing Boolliger Bands Buy Sell Strat
def buy_sell(symbol, data, news_sentiment):
    buy_list = []
    sell_list = []

    # senti_buy_list = []
    # senti_sell_list = []

    # senti_bought = False

    bought = False
    above_upper = False
    
    #flag_short = False
    #print(len(data))

    '''
    STRAT:
    Buy:
        Check previous days ago, if   (there is a day that is less than or equal to the lower band) and 
                                (current day is 105% of lower band or slope of lower line > -.1 and < .1)

                                or (slope of upper band is >.8)

                                or semtiment is postive
    
    Sell:
        If the price hits the lower band, this protects risk from bad buy/sell decisions and allows price to ride the trend
        or sell when sentiment is negative
        
    '''
    # temp_counter = 0

    BBPeriod = 14
    prev_days_threshold = 13

    buy_list = ([np.nan] * (BBPeriod + prev_days_threshold))
    sell_list = ([np.nan] * (BBPeriod + prev_days_threshold))

    # senti_buy_list = ([np.nan] * (BBPeriod + prev_days_threshold))
    # senti_sell_list = ([np.nan] * (BBPeriod + prev_days_threshold))

    #print(len(buy_list))
    #print(buy_list)
    prev_slope = 0

    date_bought = np.nan
    expection_counter = 0
    counter = 0
    for i in range(BBPeriod + prev_days_threshold, len(data)):
        prev_days = data.iloc[i-prev_days_threshold:i]
        #print(prev_days)
        below_lower = prev_days[prev_days["Close"] <= prev_days["Bot BBand"] * 1.05]
        #if(prev_days["Close"] <= prev_days["Bot BBand"]):
            #print()
        below_flag = False
        if(len(below_lower) > 0):
            below_flag = True 
        
        
        current_day = data.iloc[i]
        far_from_bottom = False
        #print(data.index[i])
        if (below_flag and current_day["Close"] > 1.05*current_day["Bot BBand"]):
            #print(below_lower)
            far_from_bottom = True
            #print(data.index[i])
        
        zero_slope = False
        
        prev_slope_days = data.iloc[i-5:i]
        #slope calc
        dates_ordinal = pd.to_datetime(prev_slope_days.index).map(dt.datetime.toordinal)
        prev_slope_days = prev_slope_days.copy()
        prev_slope_days['date_ordinal'] = dates_ordinal
        lower_slope, intercept, r_value, p_value, std_err = stats.linregress(prev_slope_days['date_ordinal'], prev_slope_days['Bot BBand'])
        
        upper_slope, intercept, r_value, p_value, std_err = stats.linregress(prev_slope_days['date_ordinal'], prev_slope_days['Top BBand'])


        
        
        if(data["Close"][i] >= data["Top BBand"][i]):
            above_upper = True
        
        sell_bband = False
        if(above_upper and bought and data["Close"][i] <= data["Sell BBand"][i] and lower_slope <= 0):
            sell_bband = True
        # prev_trend_days = data.iloc[i-7:i]
        # if date_bought in prev_trend_days.index:
        #     upper_to_lower = False
        # else:

        #     upper_to_lower_days = prev_trend_days[prev_trend_days["Close"] <= prev_trend_days["Mid BBand"]]
        #     if (bought == True and len(upper_to_lower_days) >= len(prev_trend_days) * .5):
        #         upper_to_lower = True


        #print("Date: {}...Slope:{}".format(data.index[i], slope))

        if(abs(lower_slope) <= .2 or (lower_slope > 0 and prev_slope < 0)):
            zero_slope = True
            # print("Change near 0 slope at", data.index[i-1])
        
        prev_slope = lower_slope
        


        vaderScore = 0
        date = data.index[i].to_pydatetime().date()
        try:
            vaderScore = news_sentiment[symbol, date]
        except KeyError:
            pass

        if(((below_flag and (far_from_bottom and zero_slope) and data["RSI"][i] <= 70) or vaderScore >= .25) and bought == False):
            date_bought = data.index[i]
            buy_list.append(data["Close"][i])
            sell_list.append(np.nan)
            bought = True
            # temp_counter = 0


        elif((((current_day["Close"] <= current_day["Bot BBand"] or sell_bband) and data["RSI"][i] >= 30) or vaderScore <= -.25) and bought == True):
            buy_list.append(np.nan)
            sell_list.append(data["Close"][i])

            bought = False
            above_upper = False
        else:
            
            buy_list.append(np.nan)
            sell_list.append(np.nan)

            # temp_counter += 1
            #print(temp_counter)
        
    # print("Total count:", counter)
    # print("Total expection count:", expection_counter)


    data["Buy"] = buy_list
    data["Sell"] = sell_list
    
    # data["Buy_Senti"] = senti_buy_list
    # data["Sell_Senti"] = senti_sell_list
    
    

    return data



    

def plot(data, symbol):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Buy and Sell {}'.format(symbol))
    plt.plot(data["Close"], color = "blue", alpha = .25)
    plt.plot(data["Mid BBand"], color = "orange", alpha = .75)
    plt.plot(data["Top BBand"], color = "purple", alpha = .75)
    plt.plot(data["Bot BBand"], color = "purple", alpha = .75)
    plt.plot(data["Sell BBand"], color = "red", alpha = .75)
    
    # plt.plot(data["ShortEMA"], color = "red", alpha = .5)
    # plt.plot(data["LongEMA"], color = "green", alpha = .5)
    if(not data["Buy"].isnull().all()):
        plt.scatter(data.index, data["Buy"], color = 'green', marker="^", alpha=1)
    if(not data["Sell"].isnull().all()):
        plt.scatter(data.index, data["Sell"], color = 'red', marker="v", alpha=1)
    
    # if(not data["Buy_Senti"].isnull().all()):
    #     plt.scatter(data.index, data["Buy_Senti"], color = 'blue', marker="^", alpha=1)
    # if(not data["Sell_Senti"].isnull().all()):
    #     #print(data["Sell_Senti"])
    #     plt.scatter(data.index, data["Sell_Senti"], color = 'orange', marker="v", alpha=1)

    fig.savefig('charts/{}_chart.png'.format(symbol))
    plt.close(fig)
    
    # fig = plt.figure(figsize=(8, 6))
    # plt.title('RSI {}'.format(symbol))
    # plt.plot(data["RSI"], color = "blue", alpha = 1)
    # fig.savefig('charts/{}_RSI_chart.png'.format(symbol))
    # fig.close(fig)

def get_total_profit(data):
    
    profit = 0
    i = 0 
    buyIndex = -1
    sellIndex = -1

    buyCol = data.columns.get_loc("Buy")
    closeCol = data.columns.get_loc("Close")

    while i < len(data):
        if(not pd.isnull(data.iloc[i,buyCol]) and buyIndex == -1): 
            buyIndex = i
            #print("BUYING at", data.iloc[i,buyCol])
        
        if(not pd.isnull(data.iloc[i,buyCol+1]) and buyIndex != -1):
            sellIndex = i
            #print("SELLING at", data.iloc[i,buyCol+1])
            #print("Profit:", data.iloc[sellIndex,buyCol+1] - data.iloc[buyIndex,buyCol])
            # print("=============")
            profit += (data.iloc[sellIndex,buyCol+1] - data.iloc[buyIndex,buyCol])
            
            buyIndex = -1
            sellIndex = -1
        if((i+1) == len(data) and buyIndex != -1):
            lastRow = -1
            while (pd.isnull(data.iloc[lastRow, closeCol])):
                lastRow -= 1


            profit += data.iloc[lastRow, closeCol] - data.iloc[buyIndex, buyCol]
        i+=1
    return profit


def run():
    holdings = open('data/qqq.csv').readlines()

    symbols = [holding.split(',')[2].strip() for holding in holdings][1:]
    
    totalProfit = 0
    totalEntry = 0
    totalExitProfit = 0
    news_sentiment = get_news_sentiment()
    for symbol in symbols:
     
        no_data = False
        try:
            data = get_data(symbol)
        except:
            no_data = True
        if (no_data):
            continue
        

        data = get_BBands(data)
        #data = get_EMA(data)
        data = get_RSI(data)

        data = buy_sell(symbol, data, news_sentiment)

        plot(data, symbol)

        entry = get_entry_price(data)
        totalEntry += entry

        exitProfit = get_exit_profit(data)
        totalExitProfit += exitProfit

        profit = get_total_profit(data)
        
        totalProfit += profit
        # print("PROFIT: ", profit)

        change = profit/entry
        percentChange = "{:.0%}".format(change)
        
        comparedChange = exitProfit/entry
        comparedChangepercent = "{:.0%}".format(comparedChange)

        print("{} or {} per share of {}...compared to {}".format(profit, percentChange, symbol, comparedChangepercent))
        if(change > comparedChange):
            print("===============================BBANDS STRAT WORKED===============================")
        

    totalChange = totalProfit/totalEntry
    percentChange = "{:.0%}".format(totalChange)
    print("====TOTAL PROFIT:", totalProfit)
    print("====TOTAL ENTRY:", totalEntry)
    print("====TOTAL CHANGE:", percentChange)

    woEMAProfit = totalExitProfit/totalEntry
    woEMAProfitPercent = "{:.0%}".format(woEMAProfit)

    print("========Without BBANDS========")
    print("=====TOTAL EXIT PROFIT:", totalExitProfit)
    print("=====TOTAL EXIT PROFIT:", woEMAProfitPercent)
    
    #return totalProfit, totalChange

run()


#temp = get_news_sentiment()

# symbol = "MDLZ"
# data = get_data(symbol)

# print(type(data.index[0]))
# print(type(data.index[0].to_pydatetime().date()))


# success = []
# for sellPeriod in range(5, 20):
#     for sellDev in [.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0]:
#         totalProfit, totalChange = run(sellPeriod, sellDev)
#         if (totalChange >= .4):
#             success.append((sellPeriod, sellDev, totalProfit, totalChange))

# print(success)

# symbol = "MDLZ"
# data = get_data(symbol)

# data = get_BBands(data)
# data = get_EMA(data)
# data = get_RSI(data)



# data = buy_sell(data)
# print(data)
# profit = get_total_profit(data)
# print(profit)

# plot(data, symbol)

    
