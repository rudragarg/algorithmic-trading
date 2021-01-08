import btalib
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import timedelta
from rtstock.stock import Stock
import statistics

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
    # data = pd.read_csv("data/prices/{}.csv".format(symbol))
    # data.index = data['Date'] 
    # del data['Date'] 
    # return data
    return yf.download(symbol,'2020-01-01','2020-12-31')

def get_BBands(data):
    mid, top, bot = btalib.bbands(data, period = 20, devs = 2.0)
    midSell, topSell, botSell = btalib.bbands(data, period = 20, devs = 2.0)


    data["Mid BBand"] = list(mid)
    data["Top BBand"] = list(top)
    data["Bot BBand"] = list(bot)
    #data["Sell BBand"] = list(botSell)

    return data

def get_RSI(data):
    rsi = btalib.rsi(data, period = 14)

    data["RSI"] = rsi.df

    return data

def prev_below_bottom_band(prev_days):
    below_lower = prev_days[prev_days["Close"] <= prev_days["Bot BBand"] * 1.03]

    below_flag = False
    if(len(below_lower) > 0):
        below_flag = True 

    return below_flag

def is_far_from_bottom(current_day):
    far_from_bottom = False
    if (current_day["Close"] > 1.03*current_day["Bot BBand"]):
        far_from_bottom = True
    return far_from_bottom

def slope_calc(prev_slope_days, prev_slope, slope_value = .15):
        zero_slope = False
        
        #slope calc
        dates_ordinal = pd.to_datetime(prev_slope_days.index).map(dt.datetime.toordinal)
        prev_slope_days = prev_slope_days.copy()
        prev_slope_days['date_ordinal'] = dates_ordinal
        lower_slope, intercept, r_value, p_value, std_err = stats.linregress(prev_slope_days['date_ordinal'], prev_slope_days['Bot BBand'])
        
        #CAN CHANGE THE SLOPE HERE
        if(abs(lower_slope) <= slope_value or (lower_slope > 0 and prev_slope < 0)):
            zero_slope = True
        
        return (zero_slope, lower_slope)
    
def get_vader_score(symbol, dateValue):
    vaderScore = 0
    scores = []
    news_sentiment = pd.read_csv('data/news/news.csv', index_col=[0,1])
    min_time = dt.datetime.min.time()
    dateTime = dt.datetime.combine(dateValue, min_time)
    date = dateTime.strftime("%Y-%m-%d")

    #NEED TO FIND A WAY TO ACCESS THIS DF!!!!

    # print(date)
    # print(type(date))
    # print(news_sentiment.index[0][1])
    # print(type(news_sentiment.index[0][1]))

    try:
        scores = []
        scores.append(news_sentiment.loc[(symbol, date)][0])
        days = 1
        date_time_obj = dt.datetime.strptime(date, '%Y-%m-%d')
        while len(scores) < 5 and days < 10:
            days += 1
            
            date_time_obj = date_time_obj - timedelta(days=1)
            date = date_time_obj.strftime("%Y-%m-%d")
            try:
                scores.append(news_sentiment.loc[(symbol, date)][0])
            except KeyError:
                continue
    except KeyError:
        pass

    if (len(scores) > 0):
        vaderScore = statistics.mean(scores)
    
    return vaderScore

def is_overbought(data, i):
    return data["RSI"][i] > 70

def is_oversold(data, i):
    return data["RSI"][i] < 30

def curr_below_bottom(current_day):
    return current_day["Close"] <= current_day["Bot BBand"]



#Implementing Boolliger Bands Buy Sell Strat
def buy_sell(symbol, data):
    buy_list = []
    sell_list = []


    bought = False
    

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

    BBPeriod = 14
    prev_days_threshold = 14

    buy_list = ([np.nan] * (BBPeriod + prev_days_threshold))
    sell_list = ([np.nan] * (BBPeriod + prev_days_threshold))
    prev_slope = 0

    for i in range(BBPeriod + prev_days_threshold, len(data)):
        prev_days = data.iloc[i-prev_days_threshold:i]
        # below_lower = prev_days[prev_days["Close"] <= prev_days["Bot BBand"] * 1.03]

        # below_flag = False
        # if(len(below_lower) > 0):
        #     below_flag = True 
        
        below_flag = prev_below_bottom_band(prev_days)
        
        current_day = data.iloc[i]

        far_from_bottom = is_far_from_bottom(current_day)
        
        prev_slope_days = data.iloc[i-5:i]
        slope_result = slope_calc(prev_slope_days, prev_slope)
        zero_slope = slope_result[0]
        prev_slope = slope_result[1]
        

        date = data.index[i].to_pydatetime().date()
        vaderScore = get_vader_score(symbol, date)
      
        overbought = is_overbought(data, i)
        oversold = is_oversold(data, i)

        if(((below_flag and far_from_bottom and zero_slope and not overbought) or vaderScore >= .2) and bought == False):
        
            buy_list.append(data["Close"][i])
            sell_list.append(np.nan)
            bought = True


        elif((((current_day["Close"] <= current_day["Bot BBand"]) and not oversold) or (vaderScore <= -.2)) and bought == True):
            buy_list.append(np.nan)
            sell_list.append(data["Close"][i])
            bought = False

        else:
            
            buy_list.append(np.nan)
            sell_list.append(np.nan)



    data["Buy"] = buy_list
    data["Sell"] = sell_list
    
    

    return data







def plot(data, symbol):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Buy and Sell {}'.format(symbol))
    plt.plot(data["Close"], color = "blue", alpha = .5)
    plt.plot(data["Mid BBand"], color = "orange", alpha = .75)
    plt.plot(data["Top BBand"], color = "purple", alpha = .75)
    plt.plot(data["Bot BBand"], color = "purple", alpha = .75)

    if(not data["Buy"].isnull().all()):
        plt.scatter(data.index, data["Buy"], color = 'green', marker="^", alpha=1)
    if(not data["Sell"].isnull().all()):
        plt.scatter(data.index, data["Sell"], color = 'red', marker="v", alpha=1)

    fig.savefig('TAcharts/{}_chart.png'.format(symbol))
    plt.close(fig)
    



def get_entry_price(data):
    return data.iloc[0,3]

def get_exit_profit(data):
    return data.iloc[-1, 3] - data.iloc[0,3]


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
        
        if(not pd.isnull(data.iloc[i,buyCol+1]) and buyIndex != -1):
            sellIndex = i
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


def update_news():

    news_sentiment = get_news_sentiment()

    saved_news = pd.read_csv('data/news/news.csv', index_col=[0,1])
    print(saved_news.shape)
    for index in news_sentiment.index:
        symbol = index[0]
        dateIndex = index[1]
        min_time = dt.datetime.min.time()
        dateTime = dt.datetime.combine(dateIndex, min_time)
        date = dateTime.strftime("%Y-%m-%d")

        if(not saved_news.index.isin([(symbol,date)]).any()):
            print("{} and {} does not exist".format(symbol, date))
            print(news_sentiment[index])
            saved_news.loc[(symbol, date), :] = news_sentiment[index]
        

    saved_news = saved_news.sort_index()
    saved_news.to_csv("data/news/news.csv")
    print(saved_news.shape)

def reset_news():
    news_sentiment = get_news_sentiment()
    news_sentiment.to_csv("data/news/news.csv")


def buy_sell_today(symbol, data):
    side = ""
    
    BBPeriod = 14
    prev_days_threshold = 14

    prev_slope = 0

    prev_days = data.iloc[-prev_days_threshold:]
    
    below_flag = prev_below_bottom_band(prev_days)
    current_day = data.iloc[-1]

    far_from_bottom = is_far_from_bottom(current_day)
    
    prev_slope_days = data.iloc[-5:]
    slope_result = slope_calc(prev_slope_days, prev_slope)
    zero_slope = slope_result[0]
    prev_slope = slope_result[1]
    

    date = data.index[-1].to_pydatetime().date()
    vaderScore = get_vader_score(symbol, date)
    
    overbought = is_overbought(data, -1)
    oversold = is_oversold(data, -1)

    if(((below_flag and far_from_bottom and zero_slope and not overbought) or vaderScore >= .2)):
        side = "buy"


    elif((((current_day["Close"] <= current_day["Bot BBand"]) and not oversold) or (vaderScore <= -.2))):
        side = "sell"

    else:
        
        side = "pass"



    return side




symbol = "AAPL"
data = yf.download(symbol, "2020-01-01", "2021-01-08")
data = get_BBands(data)
data = get_RSI(data)

# data = buy_sell(symbol, data)

# print(data[data["Buy"].notnull()])
# print(data[data["Sell"].notnull()])


print(data)
side = buy_sell_today(symbol, data)
print(side)
exit()


#reset_news()
news_sentiment = pd.read_csv("data/news/news.csv")
holdings = open('data/qqq.csv').readlines()

symbols = [holding.split(',')[2].strip() for holding in holdings][1:]

totalProfit = 0
totalEntry = 0
totalExitProfit = 0
countWorked = 0
for symbol in symbols:
    
    no_data = False
    try:
        data = get_data(symbol)
    except:
        no_data = True
    if (no_data):
        continue
    

    data = get_BBands(data)
    data = get_RSI(data)

    result = buy_sell(symbol, data, news_sentiment)
    #data = result[0]
    data = result

    plot(data, symbol)

    entry = get_entry_price(data)
    totalEntry += entry

    exitProfit = get_exit_profit(data)
    totalExitProfit += exitProfit

    profit = get_total_profit(data)
    
    totalProfit += profit

    change = profit/entry
    percentChange = "{:.0%}".format(change)
    
    comparedChange = exitProfit/entry
    comparedChangepercent = "{:.0%}".format(comparedChange)

    print("{} or {} per share of {}...compared to {}".format(profit, percentChange, symbol, comparedChangepercent))
    if(change >= comparedChange):
        print("===============================BBANDS STRAT WORKED===============================")
        countWorked += 1

    

totalChange = totalProfit/totalEntry
percentChange = "{:.0%}".format(totalChange)
print("====TOTAL PROFIT:", totalProfit)
print("====TOTAL ENTRY:", totalEntry)
print("====TOTAL CHANGE:", percentChange)
print("====NUM TIMES WORK: {}/{}".format(countWorked, len(symbols)))

woEMAProfit = totalExitProfit/totalEntry
woEMAProfitPercent = "{:.0%}".format(woEMAProfit)

print("========Without BBANDS========")
print("=====TOTAL EXIT PROFIT:", totalExitProfit)
print("=====TOTAL EXIT PROFIT:", woEMAProfitPercent)